//! CPU compute backend implementation.
//!
//! Implements all [`ComputeBackend`] operations using pure Rust on the CPU.
//! No unsafe code. No external C dependencies.

use tracing::{debug, trace};

use crate::tensor::{Tensor, TensorDtype, TensorStorage};

use super::{ComputeBackend, DeviceTensor};

/// CPU compute backend. All operations run on the host CPU.
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend.
    pub fn new() -> Self {
        debug!("Initialized CpuBackend");
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuBackend {
    fn upload(&self, tensor: &Tensor) -> DeviceTensor {
        trace!(shape = ?tensor.shape(), dtype = ?tensor.dtype(), "CPU upload (clone)");
        DeviceTensor::new(tensor.clone())
    }

    fn download(&self, tensor: &DeviceTensor) -> Tensor {
        trace!(shape = ?tensor.shape(), dtype = ?tensor.dtype(), "CPU download (clone)");
        tensor.tensor.clone()
    }

    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        // [M, K] x [K, N] -> [M, N]
        let a_data = a.tensor.as_f32();
        let b_data = b.tensor.as_f32();
        let a_shape = a.shape();
        let b_shape = b.shape();

        assert_eq!(a_shape.len(), 2, "matmul: a must be 2D, got shape {:?}", a_shape);
        assert_eq!(b_shape.len(), 2, "matmul: b must be 2D, got shape {:?}", b_shape);

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        assert_eq!(
            k, b_shape[0],
            "matmul: inner dimensions must match: a is [{}, {}], b is [{}, {}]",
            m, k, b_shape[0], n
        );

        trace!(m, k, n, "CPU matmul");

        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        DeviceTensor::new(Tensor::new(vec![m, n], result))
    }

    fn matmul_transpose(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        // [M, K] x [N, K]^T -> [M, N]
        // B is stored as [N, K], we treat it as transposed => [K, N]
        let a_data = a.tensor.as_f32();
        let b_data = b.tensor.as_f32();
        let a_shape = a.shape();
        let b_shape = b.shape();

        assert_eq!(a_shape.len(), 2, "matmul_transpose: a must be 2D, got shape {:?}", a_shape);
        assert_eq!(b_shape.len(), 2, "matmul_transpose: b must be 2D, got shape {:?}", b_shape);

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[0];

        assert_eq!(
            k, b_shape[1],
            "matmul_transpose: a cols ({}) must match b cols ({})",
            k, b_shape[1]
        );

        trace!(m, k, n, "CPU matmul_transpose");

        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    // a[i, p] * b[j, p] (b is [N, K], row j has the elements)
                    sum += a_data[i * k + p] * b_data[j * k + p];
                }
                result[i * n + j] = sum;
            }
        }

        DeviceTensor::new(Tensor::new(vec![m, n], result))
    }

    fn quantized_matmul(&self, weights: &DeviceTensor, input: &DeviceTensor) -> DeviceTensor {
        // Weights: [N, K] in Q8_0 format
        // Input: [M, K] in F32
        // Output: [M, N] in F32
        // This is equivalent to input x weights^T but with fused dequantization.
        let input_data = input.tensor.as_f32();
        let input_shape = input.shape();
        let weight_shape = weights.shape();

        assert_eq!(input_shape.len(), 2, "quantized_matmul: input must be 2D");
        assert_eq!(weight_shape.len(), 2, "quantized_matmul: weights must be 2D");
        assert_eq!(
            weights.dtype(),
            TensorDtype::Q8_0,
            "quantized_matmul: weights must be Q8_0, got {:?}",
            weights.dtype()
        );

        let m = input_shape[0];
        let k = input_shape[1];
        let n = weight_shape[0];

        assert_eq!(
            k, weight_shape[1],
            "quantized_matmul: input cols ({}) must match weight cols ({})",
            k, weight_shape[1]
        );

        trace!(m, k, n, "CPU quantized_matmul (Q8_0)");

        let raw = match weights.tensor.storage() {
            TensorStorage::Quantized(data) => data,
            _ => panic!("quantized_matmul: expected Quantized storage"),
        };

        // Each row of weights has K elements, packed into K/32 Q8_0 blocks
        // Each block = 34 bytes (2 bytes f16 scale + 32 bytes i8 values)
        let blocks_per_row = (k + 31) / 32;
        let bytes_per_row = blocks_per_row * 34;

        let mut result = vec![0.0f32; m * n];

        for i in 0..m {
            let input_row = &input_data[i * k..i * k + k];
            for j in 0..n {
                let row_start = j * bytes_per_row;
                let mut sum = 0.0f32;

                for block_idx in 0..blocks_per_row {
                    let block_start = row_start + block_idx * 34;
                    let scale_bits =
                        u16::from_le_bytes([raw[block_start], raw[block_start + 1]]);
                    let scale = half::f16::from_bits(scale_bits).to_f32();

                    let elem_start = block_idx * 32;
                    let elem_end = std::cmp::min(elem_start + 32, k);

                    // Fused dequant + dot product
                    for e in elem_start..elem_end {
                        let qs = raw[block_start + 2 + (e - elem_start)] as i8;
                        sum += scale * (qs as f32) * input_row[e];
                    }
                }

                result[i * n + j] = sum;
            }
        }

        DeviceTensor::new(Tensor::new(vec![m, n], result))
    }

    fn add(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let a_data = a.tensor.as_f32();
        let b_data = b.tensor.as_f32();
        assert_eq!(
            a.shape(),
            b.shape(),
            "add: shapes must match, got {:?} and {:?}",
            a.shape(),
            b.shape()
        );

        trace!(shape = ?a.shape(), "CPU add");

        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(&x, &y)| x + y).collect();
        DeviceTensor::new(Tensor::new(a.shape().to_vec(), result))
    }

    fn add_bias(&self, a: &DeviceTensor, bias: &DeviceTensor) -> DeviceTensor {
        // [M, N] + [N] -> [M, N]
        let a_data = a.tensor.as_f32();
        let bias_data = bias.tensor.as_f32();
        let a_shape = a.shape();
        let bias_shape = bias.shape();

        assert_eq!(a_shape.len(), 2, "add_bias: a must be 2D, got shape {:?}", a_shape);
        assert_eq!(bias_shape.len(), 1, "add_bias: bias must be 1D, got shape {:?}", bias_shape);

        let m = a_shape[0];
        let n = a_shape[1];
        assert_eq!(
            n,
            bias_shape[0],
            "add_bias: a cols ({}) must match bias length ({})",
            n,
            bias_shape[0]
        );

        trace!(m, n, "CPU add_bias");

        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                result[i * n + j] = a_data[i * n + j] + bias_data[j];
            }
        }

        DeviceTensor::new(Tensor::new(vec![m, n], result))
    }

    fn gelu(&self, t: &DeviceTensor) -> DeviceTensor {
        // Approximate GELU: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let data = t.tensor.as_f32();
        trace!(n_elements = data.len(), "CPU gelu");

        let sqrt_2_over_pi: f32 = (2.0f32 / std::f32::consts::PI).sqrt();

        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();

        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn silu(&self, t: &DeviceTensor) -> DeviceTensor {
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        let data = t.tensor.as_f32();
        trace!(n_elements = data.len(), "CPU silu");

        let result: Vec<f32> = data
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();

        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn swiglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        // SwiGLU: silu(gate) * up
        let gate_data = gate.tensor.as_f32();
        let up_data = up.tensor.as_f32();
        assert_eq!(
            gate.shape(),
            up.shape(),
            "swiglu: gate and up shapes must match, got {:?} and {:?}",
            gate.shape(),
            up.shape()
        );

        trace!(n_elements = gate_data.len(), "CPU swiglu");

        let result: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| {
                let silu_g = g * (1.0 / (1.0 + (-g).exp()));
                silu_g * u
            })
            .collect();

        DeviceTensor::new(Tensor::new(gate.shape().to_vec(), result))
    }

    fn layer_norm(
        &self,
        t: &DeviceTensor,
        weight: &DeviceTensor,
        bias: &DeviceTensor,
        eps: f32,
    ) -> DeviceTensor {
        // Per-row: normalize, scale by weight, add bias
        let data = t.tensor.as_f32();
        let w = weight.tensor.as_f32();
        let b = bias.tensor.as_f32();
        let shape = t.shape();

        // Works on the last dimension
        let ndim = shape.len();
        assert!(ndim >= 1, "layer_norm: tensor must be at least 1D");

        let hidden_size = shape[ndim - 1];
        let n_rows: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        assert_eq!(w.len(), hidden_size, "layer_norm: weight length must match hidden size");
        assert_eq!(b.len(), hidden_size, "layer_norm: bias length must match hidden size");

        trace!(n_rows, hidden_size, eps, "CPU layer_norm");

        let mut result = vec![0.0f32; data.len()];

        for row in 0..n_rows {
            let start = row * hidden_size;
            let end = start + hidden_size;
            let row_data = &data[start..end];

            // Compute mean
            let mean: f32 = row_data.iter().sum::<f32>() / hidden_size as f32;

            // Compute variance
            let variance: f32 =
                row_data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / hidden_size as f32;

            let inv_std = 1.0 / (variance + eps).sqrt();

            for i in 0..hidden_size {
                result[start + i] = (row_data[i] - mean) * inv_std * w[i] + b[i];
            }
        }

        DeviceTensor::new(Tensor::new(shape.to_vec(), result))
    }

    fn rms_norm(&self, t: &DeviceTensor, weight: &DeviceTensor, eps: f32) -> DeviceTensor {
        // Per-row: x * rsqrt(mean(x^2) + eps) * weight
        let data = t.tensor.as_f32();
        let w = weight.tensor.as_f32();
        let shape = t.shape();

        let ndim = shape.len();
        assert!(ndim >= 1, "rms_norm: tensor must be at least 1D");

        let hidden_size = shape[ndim - 1];
        let n_rows: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        assert_eq!(w.len(), hidden_size, "rms_norm: weight length must match hidden size");

        trace!(n_rows, hidden_size, eps, "CPU rms_norm");

        let mut result = vec![0.0f32; data.len()];

        for row in 0..n_rows {
            let start = row * hidden_size;
            let end = start + hidden_size;
            let row_data = &data[start..end];

            // mean(x^2)
            let mean_sq: f32 =
                row_data.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;

            let rms_inv = 1.0 / (mean_sq + eps).sqrt();

            for i in 0..hidden_size {
                result[start + i] = row_data[i] * rms_inv * w[i];
            }
        }

        DeviceTensor::new(Tensor::new(shape.to_vec(), result))
    }

    fn softmax(&self, t: &DeviceTensor) -> DeviceTensor {
        // Per-row softmax (last dimension)
        let data = t.tensor.as_f32();
        let shape = t.shape();

        let ndim = shape.len();
        assert!(ndim >= 1, "softmax: tensor must be at least 1D");

        let last_dim = shape[ndim - 1];
        let n_rows: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        trace!(n_rows, last_dim, "CPU softmax");

        let mut result = vec![0.0f32; data.len()];

        for row in 0..n_rows {
            let start = row * last_dim;
            let end = start + last_dim;
            let row_data = &data[start..end];

            // Numerical stability: subtract max
            let max_val = row_data
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let mut sum = 0.0f32;
            for i in 0..last_dim {
                let exp_val = (row_data[i] - max_val).exp();
                result[start + i] = exp_val;
                sum += exp_val;
            }

            if sum > 0.0 {
                for i in 0..last_dim {
                    result[start + i] /= sum;
                }
            }
        }

        DeviceTensor::new(Tensor::new(shape.to_vec(), result))
    }

    fn scale(&self, t: &DeviceTensor, factor: f32) -> DeviceTensor {
        let data = t.tensor.as_f32();
        trace!(n_elements = data.len(), factor, "CPU scale");

        let result: Vec<f32> = data.iter().map(|&x| x * factor).collect();
        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn apply_causal_mask(&self, scores: &DeviceTensor, seq_len: usize) -> DeviceTensor {
        // For attention scores [seq_len, seq_len], set positions where col > row to -inf
        let data = scores.tensor.as_f32();
        let shape = scores.shape();

        assert!(
            shape.len() >= 2,
            "apply_causal_mask: scores must be at least 2D, got shape {:?}",
            shape
        );

        trace!(?shape, seq_len, "CPU apply_causal_mask");

        let mut result = data.to_vec();
        let last_two = &shape[shape.len() - 2..];
        let rows = last_two[0];
        let cols = last_two[1];
        let batch_size: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);

        for batch in 0..batch_size {
            let batch_offset = batch * rows * cols;
            for row in 0..rows {
                for col in 0..cols {
                    if col > row {
                        result[batch_offset + row * cols + col] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        DeviceTensor::new(Tensor::new(shape.to_vec(), result))
    }

    fn rope(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
    ) -> (DeviceTensor, DeviceTensor) {
        // q, k: [seq_len, total_dim] where total_dim = n_heads * head_dim
        // Apply rotary position embeddings to each head independently.
        // For dimension pair (2i, 2i+1) within each head:
        //   freq[i] = 1.0 / (freq_base ^ (2i / head_dim))
        //   cos_theta = cos(pos * freq[i])
        //   sin_theta = sin(pos * freq[i])
        //   q_rot[2i]   = q[2i]   * cos_theta - q[2i+1] * sin_theta
        //   q_rot[2i+1] = q[2i]   * sin_theta + q[2i+1] * cos_theta

        let q_data = q.tensor.as_f32();
        let k_data = k.tensor.as_f32();
        let q_shape = q.shape();
        let k_shape = k.shape();

        assert_eq!(q_shape.len(), 2, "rope: q must be 2D, got shape {:?}", q_shape);
        assert_eq!(k_shape.len(), 2, "rope: k must be 2D, got shape {:?}", k_shape);

        let seq_len = q_shape[0];
        let total_dim = q_shape[1];

        assert_eq!(
            k_shape[0], seq_len,
            "rope: q and k must have same seq_len"
        );
        assert!(
            total_dim % head_dim == 0,
            "rope: total_dim ({}) must be divisible by head_dim ({})",
            total_dim,
            head_dim
        );
        assert!(
            head_dim % 2 == 0,
            "rope: head_dim ({}) must be even",
            head_dim
        );

        let n_heads = total_dim / head_dim;
        let k_total_dim = k_shape[1];
        let k_n_heads = k_total_dim / head_dim;

        trace!(seq_len, total_dim, head_dim, n_heads, pos_offset, freq_base, "CPU rope");

        // Precompute frequencies for one head
        let half_dim = head_dim / 2;
        let mut freqs = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            freqs[i] = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        }

        let mut q_rot = vec![0.0f32; q_data.len()];
        let mut k_rot = vec![0.0f32; k_data.len()];

        for pos in 0..seq_len {
            let abs_pos = (pos + pos_offset) as f32;

            // Apply to Q
            for head in 0..n_heads {
                let offset = pos * total_dim + head * head_dim;
                for i in 0..half_dim {
                    let theta = abs_pos * freqs[i];
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let q0 = q_data[offset + 2 * i];
                    let q1 = q_data[offset + 2 * i + 1];

                    q_rot[offset + 2 * i] = q0 * cos_theta - q1 * sin_theta;
                    q_rot[offset + 2 * i + 1] = q0 * sin_theta + q1 * cos_theta;
                }
            }

            // Apply to K
            for head in 0..k_n_heads {
                let offset = pos * k_total_dim + head * head_dim;
                for i in 0..half_dim {
                    let theta = abs_pos * freqs[i];
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let k0 = k_data[offset + 2 * i];
                    let k1 = k_data[offset + 2 * i + 1];

                    k_rot[offset + 2 * i] = k0 * cos_theta - k1 * sin_theta;
                    k_rot[offset + 2 * i + 1] = k0 * sin_theta + k1 * cos_theta;
                }
            }
        }

        (
            DeviceTensor::new(Tensor::new(q_shape.to_vec(), q_rot)),
            DeviceTensor::new(Tensor::new(k_shape.to_vec(), k_rot)),
        )
    }

    fn mean_pool(&self, hidden: &DeviceTensor, mask: &[f32]) -> DeviceTensor {
        // hidden: [seq_len, hidden_size], mask: [seq_len]
        // output: [hidden_size]
        let data = hidden.tensor.as_f32();
        let shape = hidden.shape();

        assert_eq!(shape.len(), 2, "mean_pool: hidden must be 2D, got shape {:?}", shape);

        let seq_len = shape[0];
        let hidden_size = shape[1];

        assert_eq!(
            mask.len(),
            seq_len,
            "mean_pool: mask length ({}) must match seq_len ({})",
            mask.len(),
            seq_len
        );

        trace!(seq_len, hidden_size, "CPU mean_pool");

        let mask_sum: f32 = mask.iter().sum();
        let mut result = vec![0.0f32; hidden_size];

        if mask_sum > 0.0 {
            for i in 0..seq_len {
                let m = mask[i];
                if m != 0.0 {
                    for j in 0..hidden_size {
                        result[j] += data[i * hidden_size + j] * m;
                    }
                }
            }
            for j in 0..hidden_size {
                result[j] /= mask_sum;
            }
        }

        DeviceTensor::new(Tensor::new(vec![hidden_size], result))
    }

    fn l2_normalize(&self, t: &DeviceTensor) -> DeviceTensor {
        let data = t.tensor.as_f32();
        trace!(n_elements = data.len(), "CPU l2_normalize");

        let norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();

        let result: Vec<f32> = if norm > 0.0 {
            data.iter().map(|&x| x / norm).collect()
        } else {
            data.to_vec()
        };

        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn embedding_lookup(&self, table: &DeviceTensor, ids: &[u32]) -> DeviceTensor {
        // table: [vocab_size, hidden_size]
        // ids: token IDs
        // output: [len(ids), hidden_size]
        let shape = table.shape();
        assert_eq!(
            shape.len(),
            2,
            "embedding_lookup: table must be 2D, got shape {:?}",
            shape
        );

        let vocab_size = shape[0];
        let hidden_size = shape[1];

        trace!(vocab_size, hidden_size, n_tokens = ids.len(), "CPU embedding_lookup");

        // Handle quantized embedding tables by dequantizing first
        let table_f32;
        let table_data = match table.tensor.dtype() {
            TensorDtype::F32 => table.tensor.as_f32(),
            _ => {
                table_f32 = table.tensor.to_f32();
                table_f32.as_f32()
            }
        };

        let n_tokens = ids.len();
        let mut result = vec![0.0f32; n_tokens * hidden_size];

        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(
                id < vocab_size,
                "embedding_lookup: token ID {} out of range (vocab_size={})",
                id,
                vocab_size
            );
            let src_start = id * hidden_size;
            let dst_start = i * hidden_size;
            result[dst_start..dst_start + hidden_size]
                .copy_from_slice(&table_data[src_start..src_start + hidden_size]);
        }

        DeviceTensor::new(Tensor::new(vec![n_tokens, hidden_size], result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> CpuBackend {
        CpuBackend::new()
    }

    fn dt(tensor: Tensor) -> DeviceTensor {
        DeviceTensor::new(tensor)
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", msg, a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{}: index {} differs: {} vs {} (diff={})",
                msg,
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    // ---- upload/download ----

    #[test]
    fn test_upload_download() {
        let b = backend();
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let dev = b.upload(&t);
        let back = b.download(&dev);
        assert_eq!(back.as_f32(), t.as_f32());
        assert_eq!(back.shape(), t.shape());
    }

    // ---- matmul ----

    #[test]
    fn test_matmul_identity() {
        let b = backend();
        // [2,2] x [2,2] identity
        let a = dt(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        let id = dt(Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]));
        let result = b.matmul(&a, &id);
        assert_close(result.tensor.as_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-6, "matmul identity");
    }

    #[test]
    fn test_matmul_basic() {
        let b = backend();
        // [2,3] x [3,2] -> [2,2]
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let bm = dt(Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));
        let result = b.matmul(&a, &bm);
        assert_eq!(result.shape(), &[2, 2]);
        // Row 0: 1*7+2*9+3*11 = 7+18+33=58, 1*8+2*10+3*12=8+20+36=64
        // Row 1: 4*7+5*9+6*11=28+45+66=139, 4*8+5*10+6*12=32+50+72=154
        assert_close(result.tensor.as_f32(), &[58.0, 64.0, 139.0, 154.0], 1e-5, "matmul basic");
    }

    #[test]
    fn test_matmul_single_element() {
        let b = backend();
        let a = dt(Tensor::new(vec![1, 1], vec![3.0]));
        let bm = dt(Tensor::new(vec![1, 1], vec![4.0]));
        let result = b.matmul(&a, &bm);
        assert_close(result.tensor.as_f32(), &[12.0], 1e-6, "matmul single");
    }

    #[test]
    fn test_matmul_rectangular() {
        let b = backend();
        // [1,3] x [3,1] -> [1,1]
        let a = dt(Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]));
        let bm = dt(Tensor::new(vec![3, 1], vec![4.0, 5.0, 6.0]));
        let result = b.matmul(&a, &bm);
        assert_eq!(result.shape(), &[1, 1]);
        assert_close(result.tensor.as_f32(), &[32.0], 1e-5, "matmul rect");
    }

    // ---- matmul_transpose ----

    #[test]
    fn test_matmul_transpose_basic() {
        let b = backend();
        // [2,3] x [2,3]^T -> [2,2]
        // A = [[1,2,3],[4,5,6]], B = [[7,8,9],[10,11,12]]
        // A x B^T: row 0 of A dot row 0 of B = 1*7+2*8+3*9=7+16+27=50
        //          row 0 of A dot row 1 of B = 1*10+2*11+3*12=10+22+36=68
        //          row 1 of A dot row 0 of B = 4*7+5*8+6*9=28+40+54=122
        //          row 1 of A dot row 1 of B = 4*10+5*11+6*12=40+55+72=167
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let bm = dt(Tensor::new(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]));
        let result = b.matmul_transpose(&a, &bm);
        assert_eq!(result.shape(), &[2, 2]);
        assert_close(
            result.tensor.as_f32(),
            &[50.0, 68.0, 122.0, 167.0],
            1e-5,
            "matmul_transpose",
        );
    }

    #[test]
    fn test_matmul_transpose_equals_matmul() {
        let b = backend();
        // matmul(A, B) should equal matmul_transpose(A, B^T)
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let bmat = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result_normal = b.matmul(&a, &dt(bmat.clone()));

        // B^T: transpose [3,2] -> [2,3]
        // [[7,9,11],[8,10,12]]
        let bt = dt(Tensor::new(vec![2, 3], vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]));
        let result_transpose = b.matmul_transpose(&a, &bt);

        assert_close(
            result_normal.tensor.as_f32(),
            result_transpose.tensor.as_f32(),
            1e-5,
            "matmul vs matmul_transpose",
        );
    }

    // ---- quantized_matmul ----

    #[test]
    fn test_quantized_matmul_vs_dequant() {
        let b = backend();

        // Create a 2x32 Q8_0 weight matrix (2 rows, each row is 1 block of 32)
        let scale = half::f16::from_f32(0.1);
        let mut raw = Vec::new();
        for row in 0..2u8 {
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            for i in 0..32u8 {
                raw.push(((row as i8 * 10 + i as i8) % 127) as u8);
            }
        }

        let weights_q = Tensor::from_quantized(vec![2, 32], TensorDtype::Q8_0, raw);
        let weights_f32 = weights_q.to_f32();

        // Input: [1, 32]
        let input_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let input = Tensor::new(vec![1, 32], input_data);

        // Quantized matmul
        let result_q = b.quantized_matmul(&dt(weights_q), &dt(input.clone()));

        // Reference: dequantize weights then matmul_transpose
        let result_ref = b.matmul_transpose(&dt(input), &dt(weights_f32));

        assert_close(
            result_q.tensor.as_f32(),
            result_ref.tensor.as_f32(),
            1e-4,
            "quantized_matmul vs dequant reference",
        );
    }

    #[test]
    fn test_quantized_matmul_multi_row_input() {
        let b = backend();

        // Weights: [3, 32] Q8_0 (3 output rows)
        let scale = half::f16::from_f32(0.5);
        let mut raw = Vec::new();
        for _ in 0..3 {
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            for i in 0..32u8 {
                raw.push(i);
            }
        }

        let weights_q = Tensor::from_quantized(vec![3, 32], TensorDtype::Q8_0, raw);
        let weights_f32 = weights_q.to_f32();

        // Input: [2, 32]
        let input_data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let input = Tensor::new(vec![2, 32], input_data);

        let result_q = b.quantized_matmul(&dt(weights_q), &dt(input.clone()));
        let result_ref = b.matmul_transpose(&dt(input), &dt(weights_f32));

        assert_eq!(result_q.shape(), &[2, 3]);
        assert_close(
            result_q.tensor.as_f32(),
            result_ref.tensor.as_f32(),
            1e-3,
            "quantized_matmul multi-row",
        );
    }

    // ---- add ----

    #[test]
    fn test_add() {
        let b = backend();
        let a = dt(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
        let bv = dt(Tensor::new(vec![3], vec![10.0, 20.0, 30.0]));
        let result = b.add(&a, &bv);
        assert_close(result.tensor.as_f32(), &[11.0, 22.0, 33.0], 1e-6, "add");
    }

    #[test]
    fn test_add_2d() {
        let b = backend();
        let a = dt(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        let bv = dt(Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]));
        let result = b.add(&a, &bv);
        assert_close(result.tensor.as_f32(), &[11.0, 22.0, 33.0, 44.0], 1e-6, "add 2d");
    }

    // ---- add_bias ----

    #[test]
    fn test_add_bias() {
        let b = backend();
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let bias = dt(Tensor::new(vec![3], vec![10.0, 20.0, 30.0]));
        let result = b.add_bias(&a, &bias);
        assert_eq!(result.shape(), &[2, 3]);
        assert_close(
            result.tensor.as_f32(),
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
            1e-6,
            "add_bias",
        );
    }

    // ---- gelu ----

    #[test]
    fn test_gelu_zero() {
        let b = backend();
        let t = dt(Tensor::new(vec![1], vec![0.0]));
        let result = b.gelu(&t);
        assert_close(result.tensor.as_f32(), &[0.0], 1e-6, "gelu(0)");
    }

    #[test]
    fn test_gelu_values() {
        let b = backend();
        // Known approximate GELU values:
        // gelu(1.0) ~ 0.8412
        // gelu(-1.0) ~ -0.1588
        let t = dt(Tensor::new(vec![2], vec![1.0, -1.0]));
        let result = b.gelu(&t);
        let data = result.tensor.as_f32();
        assert!((data[0] - 0.8412).abs() < 0.001, "gelu(1.0) = {}", data[0]);
        assert!((data[1] - (-0.1588)).abs() < 0.001, "gelu(-1.0) = {}", data[1]);
    }

    #[test]
    fn test_gelu_large_positive() {
        let b = backend();
        // For large x, gelu(x) ~ x
        let t = dt(Tensor::new(vec![1], vec![10.0]));
        let result = b.gelu(&t);
        assert!((result.tensor.as_f32()[0] - 10.0).abs() < 0.001, "gelu(10.0) ~ 10.0");
    }

    // ---- silu ----

    #[test]
    fn test_silu_zero() {
        let b = backend();
        let t = dt(Tensor::new(vec![1], vec![0.0]));
        let result = b.silu(&t);
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert_close(result.tensor.as_f32(), &[0.0], 1e-6, "silu(0)");
    }

    #[test]
    fn test_silu_values() {
        let b = backend();
        // silu(1.0) = 1.0 * sigmoid(1.0) = 1.0 * 0.7311 = 0.7311
        // silu(-1.0) = -1.0 * sigmoid(-1.0) = -1.0 * 0.2689 = -0.2689
        let t = dt(Tensor::new(vec![2], vec![1.0, -1.0]));
        let result = b.silu(&t);
        let data = result.tensor.as_f32();
        assert!((data[0] - 0.7311).abs() < 0.001, "silu(1.0) = {}", data[0]);
        assert!((data[1] - (-0.2689)).abs() < 0.001, "silu(-1.0) = {}", data[1]);
    }

    // ---- swiglu ----

    #[test]
    fn test_swiglu() {
        let b = backend();
        let gate = dt(Tensor::new(vec![3], vec![1.0, 0.0, -1.0]));
        let up = dt(Tensor::new(vec![3], vec![2.0, 3.0, 4.0]));
        let result = b.swiglu(&gate, &up);
        let data = result.tensor.as_f32();

        // swiglu = silu(gate) * up
        // silu(1.0) * 2.0 = 0.7311 * 2.0 = 1.4621
        // silu(0.0) * 3.0 = 0.0 * 3.0 = 0.0
        // silu(-1.0) * 4.0 = -0.2689 * 4.0 = -1.0756
        assert!((data[0] - 1.4621).abs() < 0.01, "swiglu[0] = {}", data[0]);
        assert!((data[1] - 0.0).abs() < 0.01, "swiglu[1] = {}", data[1]);
        assert!((data[2] - (-1.0756)).abs() < 0.01, "swiglu[2] = {}", data[2]);
    }

    // ---- layer_norm ----

    #[test]
    fn test_layer_norm_1d() {
        let b = backend();
        // Simple case: normalize [1, 3] with weight=[1,1] and bias=[0,0]
        let t = dt(Tensor::new(vec![1, 2], vec![1.0, 3.0]));
        let w = dt(Tensor::new(vec![2], vec![1.0, 1.0]));
        let bias = dt(Tensor::new(vec![2], vec![0.0, 0.0]));
        let result = b.layer_norm(&t, &w, &bias, 1e-5);

        let data = result.tensor.as_f32();
        // mean = 2.0, var = 1.0
        // (1 - 2) / sqrt(1 + 1e-5) = -1.0
        // (3 - 2) / sqrt(1 + 1e-5) = 1.0
        assert!((data[0] - (-1.0)).abs() < 0.01, "layer_norm[0] = {}", data[0]);
        assert!((data[1] - 1.0).abs() < 0.01, "layer_norm[1] = {}", data[1]);
    }

    #[test]
    fn test_layer_norm_with_weight_bias() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 2], vec![1.0, 3.0]));
        let w = dt(Tensor::new(vec![2], vec![2.0, 2.0]));
        let bias = dt(Tensor::new(vec![2], vec![1.0, 1.0]));
        let result = b.layer_norm(&t, &w, &bias, 1e-5);
        let data = result.tensor.as_f32();
        // normalized: [-1.0, 1.0], then * 2 + 1 = [-1.0, 3.0]
        assert!((data[0] - (-1.0)).abs() < 0.01, "ln w/b [0] = {}", data[0]);
        assert!((data[1] - 3.0).abs() < 0.01, "ln w/b [1] = {}", data[1]);
    }

    #[test]
    fn test_layer_norm_multi_row() {
        let b = backend();
        // Two rows, each normalized independently
        let t = dt(Tensor::new(vec![2, 2], vec![1.0, 3.0, 10.0, 20.0]));
        let w = dt(Tensor::new(vec![2], vec![1.0, 1.0]));
        let bias = dt(Tensor::new(vec![2], vec![0.0, 0.0]));
        let result = b.layer_norm(&t, &w, &bias, 1e-5);
        let data = result.tensor.as_f32();

        // Row 0: mean=2, var=1 -> [-1, 1]
        assert!((data[0] - (-1.0)).abs() < 0.01);
        assert!((data[1] - 1.0).abs() < 0.01);
        // Row 1: mean=15, var=25 -> [-1, 1]
        assert!((data[2] - (-1.0)).abs() < 0.01);
        assert!((data[3] - 1.0).abs() < 0.01);
    }

    // ---- rms_norm ----

    #[test]
    fn test_rms_norm_basic() {
        let b = backend();
        // [1, 4] with weight=[1,1,1,1]
        let t = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let w = dt(Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
        let result = b.rms_norm(&t, &w, 1e-5);
        let data = result.tensor.as_f32();

        // mean(x^2) = (1+4+9+16)/4 = 30/4 = 7.5
        // rms = sqrt(7.5 + 1e-5) ~ 2.7386
        // result[i] = x[i] / rms
        let rms = (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (data[i] - expected).abs() < 1e-4,
                "rms_norm[{}] = {}, expected {}",
                i,
                data[i],
                expected
            );
        }
    }

    #[test]
    fn test_rms_norm_with_weight() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 2], vec![3.0, 4.0]));
        let w = dt(Tensor::new(vec![2], vec![2.0, 0.5]));
        let result = b.rms_norm(&t, &w, 1e-5);
        let data = result.tensor.as_f32();

        // mean(x^2) = (9+16)/2 = 12.5
        // rms_inv = 1/sqrt(12.5 + 1e-5)
        let rms_inv = 1.0 / (12.5f32 + 1e-5).sqrt();
        assert!((data[0] - 3.0 * rms_inv * 2.0).abs() < 1e-4);
        assert!((data[1] - 4.0 * rms_inv * 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_multi_row() {
        let b = backend();
        let t = dt(Tensor::new(vec![2, 2], vec![1.0, 1.0, 2.0, 2.0]));
        let w = dt(Tensor::new(vec![2], vec![1.0, 1.0]));
        let result = b.rms_norm(&t, &w, 0.0);
        let data = result.tensor.as_f32();

        // Row 0: mean(1^2 + 1^2)/2 = 1.0, rms=1.0, output=[1,1]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
        // Row 1: mean(4+4)/2 = 4.0, rms=2.0, output=[1,1]
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[3] - 1.0).abs() < 1e-5);
    }

    // ---- softmax ----

    #[test]
    fn test_softmax_uniform() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 3], vec![1.0, 1.0, 1.0]));
        let result = b.softmax(&t);
        let data = result.tensor.as_f32();
        // All equal => uniform distribution
        for &v in data {
            assert!((v - 1.0 / 3.0).abs() < 1e-5, "softmax uniform: {}", v);
        }
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let result = b.softmax(&t);
        let data = result.tensor.as_f32();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}", sum);
    }

    #[test]
    fn test_softmax_multi_row() {
        let b = backend();
        let t = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]));
        let result = b.softmax(&t);
        let data = result.tensor.as_f32();

        // Each row should sum to 1
        let sum_row0: f32 = data[0..3].iter().sum();
        let sum_row1: f32 = data[3..6].iter().sum();
        assert!((sum_row0 - 1.0).abs() < 1e-5, "softmax row0 sum = {}", sum_row0);
        assert!((sum_row1 - 1.0).abs() < 1e-5, "softmax row1 sum = {}", sum_row1);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let b = backend();
        // Large values should not overflow
        let t = dt(Tensor::new(vec![1, 3], vec![1000.0, 1001.0, 1002.0]));
        let result = b.softmax(&t);
        let data = result.tensor.as_f32();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax large values sum = {}", sum);
        assert!(data.iter().all(|&v| v.is_finite()), "softmax produced non-finite values");
    }

    #[test]
    fn test_softmax_with_neg_infinity() {
        let b = backend();
        // Masked positions (neg infinity) should get zero probability
        let t = dt(Tensor::new(
            vec![1, 3],
            vec![1.0, f32::NEG_INFINITY, 2.0],
        ));
        let result = b.softmax(&t);
        let data = result.tensor.as_f32();
        assert!((data[1] - 0.0).abs() < 1e-6, "masked position should be 0");
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax with mask sum = {}", sum);
    }

    // ---- scale ----

    #[test]
    fn test_scale() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
        let result = b.scale(&t, 2.5);
        assert_close(result.tensor.as_f32(), &[2.5, 5.0, 7.5], 1e-6, "scale");
    }

    #[test]
    fn test_scale_zero() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
        let result = b.scale(&t, 0.0);
        assert_close(result.tensor.as_f32(), &[0.0, 0.0, 0.0], 1e-6, "scale by 0");
    }

    // ---- apply_causal_mask ----

    #[test]
    fn test_causal_mask() {
        let b = backend();
        let scores = dt(Tensor::new(
            vec![3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ));
        let result = b.apply_causal_mask(&scores, 3);
        let data = result.tensor.as_f32();

        // Row 0: [1.0, -inf, -inf]
        // Row 1: [4.0, 5.0, -inf]
        // Row 2: [7.0, 8.0, 9.0]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 5.0);
        assert_eq!(data[5], f32::NEG_INFINITY);
        assert_eq!(data[6], 7.0);
        assert_eq!(data[7], 8.0);
        assert_eq!(data[8], 9.0);
    }

    #[test]
    fn test_causal_mask_1x1() {
        let b = backend();
        let scores = dt(Tensor::new(vec![1, 1], vec![5.0]));
        let result = b.apply_causal_mask(&scores, 1);
        assert_eq!(result.tensor.as_f32()[0], 5.0);
    }

    // ---- rope ----

    #[test]
    fn test_rope_zero_position() {
        let b = backend();
        // At position 0, cos(0)=1, sin(0)=0, so RoPE should be identity
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![5.0, 6.0, 7.0, 8.0]));
        let (q_rot, k_rot) = b.rope(&q, &k, 0, 10000.0, 4);
        assert_close(q_rot.tensor.as_f32(), &[1.0, 2.0, 3.0, 4.0], 1e-5, "rope q at pos 0");
        assert_close(k_rot.tensor.as_f32(), &[5.0, 6.0, 7.0, 8.0], 1e-5, "rope k at pos 0");
    }

    #[test]
    fn test_rope_rotation() {
        let b = backend();
        // At position 1 with head_dim=2, freq_base=1.0 (so freq=1.0):
        // theta = 1.0 * 1.0 = 1.0
        // cos(1) ~ 0.5403, sin(1) ~ 0.8415
        // q_rot[0] = q[0]*cos - q[1]*sin = 1*0.5403 - 0*0.8415 = 0.5403
        // q_rot[1] = q[0]*sin + q[1]*cos = 1*0.8415 + 0*0.5403 = 0.8415
        let q = dt(Tensor::new(vec![1, 2], vec![1.0, 0.0]));
        let k = dt(Tensor::new(vec![1, 2], vec![1.0, 0.0]));
        let (q_rot, _k_rot) = b.rope(&q, &k, 1, 1.0, 2);

        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        let data = q_rot.tensor.as_f32();
        assert!((data[0] - cos1).abs() < 1e-5, "rope rot q[0] = {}", data[0]);
        assert!((data[1] - sin1).abs() < 1e-5, "rope rot q[1] = {}", data[1]);
    }

    #[test]
    fn test_rope_preserves_norm() {
        let b = backend();
        // RoPE is a rotation, so the L2 norm should be preserved
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![5.0, 6.0, 7.0, 8.0]));
        let (q_rot, _) = b.rope(&q, &k, 5, 10000.0, 4);

        let norm_before: f32 = q.tensor.as_f32().iter().map(|&x| x * x).sum::<f32>().sqrt();
        let norm_after: f32 = q_rot.tensor.as_f32().iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE changed norm: {} -> {}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_rope_multi_head() {
        let b = backend();
        // 2 heads, head_dim=2, seq_len=1
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]));
        let (q_rot, _) = b.rope(&q, &k, 1, 1.0, 2);
        let data = q_rot.tensor.as_f32();

        // Head 0: [1,0] rotated by theta=1 -> [cos1, sin1]
        // Head 1: [0,1] rotated by theta=1 -> [-sin1, cos1]
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!((data[0] - cos1).abs() < 1e-5);
        assert!((data[1] - sin1).abs() < 1e-5);
        assert!((data[2] - (-sin1)).abs() < 1e-5);
        assert!((data[3] - cos1).abs() < 1e-5);
    }

    #[test]
    fn test_rope_multi_position() {
        let b = backend();
        // seq_len=2, head_dim=2
        let q = dt(Tensor::new(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]));
        let k = dt(Tensor::new(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]));
        let (q_rot, _) = b.rope(&q, &k, 0, 1.0, 2);
        let data = q_rot.tensor.as_f32();

        // Pos 0: theta=0, cos=1, sin=0 -> [1, 0]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        // Pos 1: theta=1, cos=cos(1), sin=sin(1) -> [cos1, sin1]
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!((data[2] - cos1).abs() < 1e-5);
        assert!((data[3] - sin1).abs() < 1e-5);
    }

    // ---- mean_pool ----

    #[test]
    fn test_mean_pool_all_ones_mask() {
        let b = backend();
        let hidden = dt(Tensor::new(
            vec![3, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let mask = vec![1.0, 1.0, 1.0];
        let result = b.mean_pool(&hidden, &mask);
        assert_eq!(result.shape(), &[2]);
        // Mean: [(1+3+5)/3, (2+4+6)/3] = [3, 4]
        assert_close(result.tensor.as_f32(), &[3.0, 4.0], 1e-5, "mean_pool all ones");
    }

    #[test]
    fn test_mean_pool_partial_mask() {
        let b = backend();
        let hidden = dt(Tensor::new(
            vec![3, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let mask = vec![1.0, 1.0, 0.0]; // Only first two tokens
        let result = b.mean_pool(&hidden, &mask);
        // Mean: [(1+3)/2, (2+4)/2] = [2, 3]
        assert_close(result.tensor.as_f32(), &[2.0, 3.0], 1e-5, "mean_pool partial mask");
    }

    #[test]
    fn test_mean_pool_zero_mask() {
        let b = backend();
        let hidden = dt(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        let mask = vec![0.0, 0.0];
        let result = b.mean_pool(&hidden, &mask);
        assert_close(result.tensor.as_f32(), &[0.0, 0.0], 1e-5, "mean_pool zero mask");
    }

    // ---- l2_normalize ----

    #[test]
    fn test_l2_normalize() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![3.0, 4.0, 0.0]));
        let result = b.l2_normalize(&t);
        let data = result.tensor.as_f32();
        // norm = 5
        assert_close(data, &[0.6, 0.8, 0.0], 1e-5, "l2_normalize");
    }

    #[test]
    fn test_l2_normalize_unit_vector() {
        let b = backend();
        let t = dt(Tensor::new(vec![2], vec![0.6, 0.8]));
        let result = b.l2_normalize(&t);
        let data = result.tensor.as_f32();
        let norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "normalized vector should have norm 1, got {}", norm);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![0.0, 0.0, 0.0]));
        let result = b.l2_normalize(&t);
        assert_close(result.tensor.as_f32(), &[0.0, 0.0, 0.0], 1e-6, "l2_normalize zero");
    }

    // ---- embedding_lookup ----

    #[test]
    fn test_embedding_lookup() {
        let b = backend();
        // Vocab of 4 tokens, hidden_size=3
        let table = dt(Tensor::new(
            vec![4, 3],
            vec![
                0.1, 0.2, 0.3, // token 0
                1.1, 1.2, 1.3, // token 1
                2.1, 2.2, 2.3, // token 2
                3.1, 3.2, 3.3, // token 3
            ],
        ));
        let ids = &[2, 0, 3];
        let result = b.embedding_lookup(&table, ids);
        assert_eq!(result.shape(), &[3, 3]);
        assert_close(
            result.tensor.as_f32(),
            &[2.1, 2.2, 2.3, 0.1, 0.2, 0.3, 3.1, 3.2, 3.3],
            1e-6,
            "embedding_lookup",
        );
    }

    #[test]
    fn test_embedding_lookup_single() {
        let b = backend();
        let table = dt(Tensor::new(
            vec![3, 2],
            vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1],
        ));
        let result = b.embedding_lookup(&table, &[1]);
        assert_eq!(result.shape(), &[1, 2]);
        assert_close(result.tensor.as_f32(), &[1.0, 1.1], 1e-6, "embedding single");
    }

    #[test]
    fn test_embedding_lookup_repeated() {
        let b = backend();
        let table = dt(Tensor::new(
            vec![2, 2],
            vec![10.0, 20.0, 30.0, 40.0],
        ));
        let result = b.embedding_lookup(&table, &[0, 0, 1, 0]);
        assert_eq!(result.shape(), &[4, 2]);
        assert_close(
            result.tensor.as_f32(),
            &[10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 10.0, 20.0],
            1e-6,
            "embedding repeated",
        );
    }

    // ---- Integration / combined tests ----

    #[test]
    fn test_softmax_after_causal_mask() {
        let b = backend();
        // Simulate attention: apply mask then softmax
        let scores = dt(Tensor::new(
            vec![3, 3],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ));
        let masked = b.apply_causal_mask(&scores, 3);
        let probs = b.softmax(&masked);
        let data = probs.tensor.as_f32();

        // Row 0: only pos 0 visible -> [1.0, 0, 0]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1]).abs() < 1e-5);
        assert!((data[2]).abs() < 1e-5);

        // Row 1: pos 0,1 visible -> [0.5, 0.5, 0]
        assert!((data[3] - 0.5).abs() < 1e-5);
        assert!((data[4] - 0.5).abs() < 1e-5);
        assert!((data[5]).abs() < 1e-5);

        // Row 2: all visible -> [1/3, 1/3, 1/3]
        for i in 6..9 {
            assert!((data[i] - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_matmul_then_add_bias() {
        let b = backend();
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        let weights = dt(Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let bias = dt(Tensor::new(vec![2], vec![10.0, 20.0]));

        let mm = b.matmul(&a, &weights);
        let result = b.add_bias(&mm, &bias);

        // Row 0: [1,2] + [10,20] = [11, 22]
        // Row 1: [3,4] + [10,20] = [13, 24]
        assert_close(
            result.tensor.as_f32(),
            &[11.0, 22.0, 13.0, 24.0],
            1e-5,
            "matmul + bias",
        );
    }

    #[test]
    fn test_rms_norm_then_swiglu() {
        // Simulate a Gemma-style FFN: rms_norm -> gate/up projections -> swiglu -> down
        let b = backend();
        let x = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let norm_weight = dt(Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]));

        let normed = b.rms_norm(&x, &norm_weight, 1e-5);

        // Verify RMS norm output is reasonable
        let norm_data = normed.tensor.as_f32();
        let sum_sq: f32 = norm_data.iter().map(|&x| x * x).sum::<f32>() / 4.0;
        assert!(
            (sum_sq - 1.0).abs() < 0.01,
            "RMS norm output should have mean(x^2) ~ 1.0, got {}",
            sum_sq
        );
    }

    #[test]
    fn test_silu_1d() {
        let b = backend();
        // Test a range of values
        let vals = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let t = dt(Tensor::new(vec![7], vals.clone()));
        let result = b.silu(&t);
        let data = result.tensor.as_f32();

        for (i, &x) in vals.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            assert!(
                (data[i] - expected).abs() < 1e-5,
                "silu({}) = {}, expected {}",
                x,
                data[i],
                expected
            );
        }
    }

    #[test]
    fn test_gelu_properties() {
        let b = backend();
        // GELU properties:
        // 1. gelu(0) = 0
        // 2. gelu(x) ~ x for large positive x
        // 3. gelu(x) ~ 0 for large negative x
        // 4. gelu(x) + gelu(-x) ~ x for the identity property
        let t = dt(Tensor::new(vec![5], vec![-5.0, -1.0, 0.0, 1.0, 5.0]));
        let result = b.gelu(&t);
        let data = result.tensor.as_f32();

        // gelu(-5) should be close to 0
        assert!(data[0].abs() < 0.01, "gelu(-5) should be ~0, got {}", data[0]);
        // gelu(0) = 0
        assert!((data[2]).abs() < 1e-6, "gelu(0) should be 0, got {}", data[2]);
        // gelu(5) should be close to 5
        assert!((data[4] - 5.0).abs() < 0.01, "gelu(5) should be ~5, got {}", data[4]);
        // For positive inputs, gelu should be monotonically increasing
        assert!(data[3] < data[4], "gelu should increase for positive values");
    }

    // ====================================================================
    // NEW TESTS: CPU backend edge cases and chained operations
    // ====================================================================

    // -- Default trait --

    #[test]
    fn test_cpu_backend_default() {
        let b = CpuBackend::default();
        let t = Tensor::new(vec![1], vec![42.0]);
        let dev = b.upload(&t);
        let back = b.download(&dev);
        assert_eq!(back.as_f32(), &[42.0]);
    }

    // -- DeviceTensor properties --

    #[test]
    fn test_device_tensor_shape_and_dtype() {
        let t = Tensor::new(vec![2, 3], vec![0.0; 6]);
        let dev = DeviceTensor::new(t);
        assert_eq!(dev.shape(), &[2, 3]);
        assert_eq!(dev.dtype(), TensorDtype::F32);
    }

    // -- matmul with larger matrices --

    #[test]
    fn test_matmul_larger() {
        let b = backend();
        // [4, 3] x [3, 5] -> [4, 5]
        let a_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..15).map(|i| (i as f32) * 0.1).collect();
        let a = dt(Tensor::new(vec![4, 3], a_data.clone()));
        let bm = dt(Tensor::new(vec![3, 5], b_data.clone()));
        let result = b.matmul(&a, &bm);
        assert_eq!(result.shape(), &[4, 5]);

        // Manually compute first element: a[0,0]*b[0,0] + a[0,1]*b[1,0] + a[0,2]*b[2,0]
        // = 0*0 + 1*0.5 + 2*1.0 = 2.5
        let data = result.tensor.as_f32();
        let expected_00 = a_data[0] * b_data[0] + a_data[1] * b_data[5] + a_data[2] * b_data[10];
        assert!((data[0] - expected_00).abs() < 1e-5, "larger matmul[0,0]");
    }

    // -- Chained operations: rms_norm -> matmul -> swiglu -> add --

    #[test]
    fn test_chained_rms_norm_matmul_swiglu_add() {
        let b = backend();
        // Input: [1, 4]
        let x = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let norm_weight = dt(Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]));

        // Step 1: RMS norm
        let normed = b.rms_norm(&x, &norm_weight, 1e-5);
        let norm_data = normed.tensor.as_f32().to_vec();

        // Verify RMS norm correctness
        let mean_sq: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x * x).sum::<f32>() / 4.0;
        let rms_inv = 1.0 / (mean_sq + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i as f32 + 1.0) * rms_inv;
            assert!((norm_data[i] - expected).abs() < 1e-4, "rms_norm at {}", i);
        }

        // Step 2: Gate and up projections (identity-like weights)
        let gate_weights = dt(Tensor::new(
            vec![4, 4],
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        ));
        let gate = b.matmul(&normed, &gate_weights);
        let up = b.matmul(&normed, &gate_weights);

        // Step 3: SwiGLU
        let swiglu_result = b.swiglu(&gate, &up);
        let swiglu_data = swiglu_result.tensor.as_f32();

        // Verify: swiglu(x, x) = silu(x) * x
        for i in 0..4 {
            let x = norm_data[i];
            let silu_x = x / (1.0 + (-x).exp());
            let expected = silu_x * x;
            assert!(
                (swiglu_data[i] - expected).abs() < 1e-4,
                "swiglu at {}: got {}, expected {}",
                i, swiglu_data[i], expected
            );
        }

        // Step 4: Add residual (original x)
        let residual = b.add(&swiglu_result, &x);
        assert_eq!(residual.shape(), &[1, 4]);
        // Just verify it runs without error and produces finite values
        for &v in residual.tensor.as_f32() {
            assert!(v.is_finite(), "chained ops produced non-finite value");
        }
    }

    // -- 1x1 matrix operations (boundary conditions) --

    #[test]
    fn test_matmul_transpose_1x1() {
        let b = backend();
        let a = dt(Tensor::new(vec![1, 1], vec![3.0]));
        let bm = dt(Tensor::new(vec![1, 1], vec![4.0]));
        let result = b.matmul_transpose(&a, &bm);
        assert_close(result.tensor.as_f32(), &[12.0], 1e-6, "matmul_transpose 1x1");
    }

    // -- Layer norm with constant input --

    #[test]
    fn test_layer_norm_constant_input() {
        let b = backend();
        // All elements the same: variance = 0, so normalized output is 0*weight + bias = bias
        let t = dt(Tensor::new(vec![1, 4], vec![5.0, 5.0, 5.0, 5.0]));
        let w = dt(Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
        let bias = dt(Tensor::new(vec![4], vec![0.5, 0.5, 0.5, 0.5]));
        let result = b.layer_norm(&t, &w, &bias, 1e-5);
        let data = result.tensor.as_f32();
        // (5.0 - 5.0) / sqrt(0 + 1e-5) * 1.0 + 0.5 = 0.5
        for &v in data {
            assert!((v - 0.5).abs() < 0.01, "constant input layer_norm: {}", v);
        }
    }

    // -- RMS norm with constant input --

    #[test]
    fn test_rms_norm_constant_input() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 3], vec![2.0, 2.0, 2.0]));
        let w = dt(Tensor::new(vec![3], vec![1.0, 1.0, 1.0]));
        let result = b.rms_norm(&t, &w, 0.0);
        let data = result.tensor.as_f32();
        // mean(x^2) = 4.0, rms = 2.0, output = 2.0/2.0 = 1.0
        for &v in data {
            assert!((v - 1.0).abs() < 1e-5, "constant rms_norm: {}", v);
        }
    }

    // -- Softmax with single element --

    #[test]
    fn test_softmax_single_element() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 1], vec![42.0]));
        let result = b.softmax(&t);
        assert!((result.tensor.as_f32()[0] - 1.0).abs() < 1e-6, "softmax of single element");
    }

    // -- Scale with negative factor --

    #[test]
    fn test_scale_negative() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![1.0, -2.0, 3.0]));
        let result = b.scale(&t, -1.0);
        assert_close(result.tensor.as_f32(), &[-1.0, 2.0, -3.0], 1e-6, "scale by -1");
    }

    // -- Causal mask on non-square matrix --

    #[test]
    fn test_causal_mask_2x3() {
        let b = backend();
        let scores = dt(Tensor::new(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));
        let result = b.apply_causal_mask(&scores, 2);
        let data = result.tensor.as_f32();
        // Row 0: [1.0, -inf, -inf]
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], f32::NEG_INFINITY);
        // Row 1: [4.0, 5.0, -inf]
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 5.0);
        assert_eq!(data[5], f32::NEG_INFINITY);
    }

    // -- RoPE with large head_dim --

    #[test]
    fn test_rope_head_dim_8() {
        let b = backend();
        // seq_len=1, 2 heads of dim 8 = total_dim 16
        let q_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.2).collect();
        let q = dt(Tensor::new(vec![1, 16], q_data));
        let k = dt(Tensor::new(vec![1, 16], k_data));
        let (q_rot, k_rot) = b.rope(&q, &k, 3, 10000.0, 8);

        // Verify output shapes
        assert_eq!(q_rot.shape(), &[1, 16]);
        assert_eq!(k_rot.shape(), &[1, 16]);

        // RoPE is a rotation, so norms should be preserved per head.
        // Head 0: elements 0..8
        let q_norm_before: f32 = (0..8).map(|i| (i as f32 * 0.1).powi(2)).sum::<f32>().sqrt();
        let q_norm_after: f32 = q_rot.tensor.as_f32()[0..8]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "RoPE should preserve head norm"
        );
    }

    // -- RoPE with different Q and K head counts (GQA) --

    #[test]
    fn test_rope_different_qk_dims() {
        let b = backend();
        // Q has 4 heads, K has 2 heads (GQA), head_dim=2
        let q = dt(Tensor::new(vec![1, 8], vec![1.0; 8])); // 4 heads * 2 dim
        let k = dt(Tensor::new(vec![1, 4], vec![1.0; 4])); // 2 heads * 2 dim
        let (q_rot, k_rot) = b.rope(&q, &k, 1, 10000.0, 2);
        assert_eq!(q_rot.shape(), &[1, 8]);
        assert_eq!(k_rot.shape(), &[1, 4]);
    }

    // -- Mean pool with single token --

    #[test]
    fn test_mean_pool_single_token() {
        let b = backend();
        let hidden = dt(Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]));
        let mask = vec![1.0];
        let result = b.mean_pool(&hidden, &mask);
        assert_close(result.tensor.as_f32(), &[1.0, 2.0, 3.0], 1e-5, "single token mean_pool");
    }

    // -- L2 normalize already-unit vector --

    #[test]
    fn test_l2_normalize_already_normalized() {
        let b = backend();
        // Already unit norm
        let t = dt(Tensor::new(vec![2], vec![0.6, 0.8]));
        let result = b.l2_normalize(&t);
        let data = result.tensor.as_f32();
        assert!((data[0] - 0.6).abs() < 1e-5);
        assert!((data[1] - 0.8).abs() < 1e-5);
    }

    // -- L2 normalize single element --

    #[test]
    fn test_l2_normalize_single() {
        let b = backend();
        let t = dt(Tensor::new(vec![1], vec![5.0]));
        let result = b.l2_normalize(&t);
        // 5.0 / |5.0| = 1.0
        assert!((result.tensor.as_f32()[0] - 1.0).abs() < 1e-6);
    }

    // -- Embedding lookup with quantized table --

    #[test]
    fn test_embedding_lookup_quantized() {
        let b = backend();
        // Create a Q8_0 embedding table: 3 tokens, hidden_size=32 (1 block per row)
        let scale = half::f16::from_f32(0.5);
        let mut raw = Vec::new();
        for row in 0..3u8 {
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            for i in 0..32u8 {
                raw.push(((row * 10 + i) % 127) as u8);
            }
        }
        let table = DeviceTensor::new(
            Tensor::from_quantized(vec![3, 32], TensorDtype::Q8_0, raw),
        );
        let result = b.embedding_lookup(&table, &[1, 0]);
        assert_eq!(result.shape(), &[2, 32]);
        // Verify it returns finite values
        for &v in result.tensor.as_f32() {
            assert!(v.is_finite(), "quantized embedding produced non-finite");
        }
    }

    // -- Embedding lookup empty ids --

    #[test]
    fn test_embedding_lookup_empty_ids() {
        let b = backend();
        let table = dt(Tensor::new(vec![3, 2], vec![0.0; 6]));
        let result = b.embedding_lookup(&table, &[]);
        assert_eq!(result.shape(), &[0, 2]);
        assert!(result.tensor.as_f32().is_empty());
    }

    // -- Silu symmetry --

    #[test]
    fn test_silu_symmetry() {
        let b = backend();
        // silu(-x) = -x * sigmoid(-x) = -x * (1 - sigmoid(x))
        // silu(x) + silu(-x) != 0 (it's not odd), but let's verify the math.
        let t_pos = dt(Tensor::new(vec![1], vec![2.0]));
        let t_neg = dt(Tensor::new(vec![1], vec![-2.0]));
        let pos = b.silu(&t_pos).tensor.as_f32()[0];
        let neg = b.silu(&t_neg).tensor.as_f32()[0];
        // silu(2) = 2 * sigmoid(2) = 2 * 0.8808 = 1.7616
        // silu(-2) = -2 * sigmoid(-2) = -2 * 0.1192 = -0.2384
        assert!((pos - 1.7616).abs() < 0.01, "silu(2) = {}", pos);
        assert!((neg - (-0.2384)).abs() < 0.01, "silu(-2) = {}", neg);
    }

    // -- SwiGLU with zero gate --

    #[test]
    fn test_swiglu_zero_gate() {
        let b = backend();
        let gate = dt(Tensor::new(vec![3], vec![0.0, 0.0, 0.0]));
        let up = dt(Tensor::new(vec![3], vec![100.0, 200.0, 300.0]));
        let result = b.swiglu(&gate, &up);
        // silu(0) = 0, so output is all zeros
        assert_close(result.tensor.as_f32(), &[0.0, 0.0, 0.0], 1e-6, "swiglu zero gate");
    }

    // -- Softmax then matmul (attention pattern) --

    #[test]
    fn test_softmax_then_matmul_attention() {
        let b = backend();
        // Simulate: attention_probs = softmax(scores), output = attention_probs x V
        let scores = dt(Tensor::new(vec![2, 2], vec![10.0, 0.0, 0.0, 10.0]));
        let probs = b.softmax(&scores);
        // With these scores, probs should be approximately [[1,0],[0,1]] (identity)

        let v = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let output = b.matmul(&probs, &v);
        assert_eq!(output.shape(), &[2, 3]);
        let data = output.tensor.as_f32();
        // Row 0 should be approximately V[0] = [1, 2, 3]
        assert!((data[0] - 1.0).abs() < 0.01);
        assert!((data[1] - 2.0).abs() < 0.01);
        assert!((data[2] - 3.0).abs() < 0.01);
        // Row 1 should be approximately V[1] = [4, 5, 6]
        assert!((data[3] - 4.0).abs() < 0.01);
        assert!((data[4] - 5.0).abs() < 0.01);
        assert!((data[5] - 6.0).abs() < 0.01);
    }

    // -- Layer norm then gelu (BERT-style) --

    #[test]
    fn test_layer_norm_then_gelu() {
        let b = backend();
        let x = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let w = dt(Tensor::new(vec![4], vec![1.0, 1.0, 1.0, 1.0]));
        let bias = dt(Tensor::new(vec![4], vec![0.0, 0.0, 0.0, 0.0]));

        let normed = b.layer_norm(&x, &w, &bias, 1e-5);
        let activated = b.gelu(&normed);

        // Verify all outputs are finite
        for &v in activated.tensor.as_f32() {
            assert!(v.is_finite(), "layer_norm->gelu produced non-finite");
        }
    }

    // -- Embedding lookup -> layer_norm -> mean_pool pipeline --

    #[test]
    fn test_embed_norm_pool_pipeline() {
        let b = backend();
        let table = dt(Tensor::new(
            vec![4, 3],
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0,
            ],
        ));
        let ids = &[0, 1, 2];
        let embedded = b.embedding_lookup(&table, ids);
        assert_eq!(embedded.shape(), &[3, 3]);

        let w = dt(Tensor::new(vec![3], vec![1.0, 1.0, 1.0]));
        let bias = dt(Tensor::new(vec![3], vec![0.0, 0.0, 0.0]));
        let normed = b.layer_norm(&embedded, &w, &bias, 1e-5);

        let mask = vec![1.0, 1.0, 1.0];
        let pooled = b.mean_pool(&normed, &mask);
        assert_eq!(pooled.shape(), &[3]);

        let normalized = b.l2_normalize(&pooled);
        let norm: f32 = normalized.tensor.as_f32().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "l2 norm should be 1.0, got {}", norm);
    }

    // -- Send + Sync --

    #[test]
    fn test_cpu_backend_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CpuBackend>();
    }

    // -- add_bias 2D with multiple rows --

    #[test]
    fn test_add_bias_large() {
        let b = backend();
        let rows = 4;
        let cols = 3;
        let a_data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let bias_data: Vec<f32> = vec![100.0, 200.0, 300.0];
        let a = dt(Tensor::new(vec![rows, cols], a_data.clone()));
        let bias = dt(Tensor::new(vec![cols], bias_data.clone()));
        let result = b.add_bias(&a, &bias);
        let data = result.tensor.as_f32();
        for i in 0..rows {
            for j in 0..cols {
                let expected = a_data[i * cols + j] + bias_data[j];
                assert!(
                    (data[i * cols + j] - expected).abs() < 1e-6,
                    "add_bias[{},{}]: got {}, expected {}",
                    i, j, data[i * cols + j], expected
                );
            }
        }
    }

    // -- Quantized matmul with identity-like weights --

    #[test]
    fn test_quantized_matmul_identity_like() {
        let b = backend();
        // Create a "nearly identity" Q8_0 weight: 1 output neuron, 32 input dims.
        // Weights = all ones (quantized as scale=1/127 * 127 = 1.0).
        let scale = half::f16::from_f32(1.0 / 127.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        for _ in 0..32 {
            raw.push(127u8); // 127 as i8 -> dequantized = (1/127)*127 ~ 1.0
        }

        let weights = Tensor::from_quantized(vec![1, 32], TensorDtype::Q8_0, raw);
        let input_data: Vec<f32> = vec![1.0; 32];
        let input = Tensor::new(vec![1, 32], input_data);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        assert_eq!(result.shape(), &[1, 1]);
        // Sum of 32 ones = 32.0 (within f16 precision)
        let val = result.tensor.as_f32()[0];
        assert!((val - 32.0).abs() < 0.5, "identity-like Q8_0 matmul: got {}", val);
    }
}
