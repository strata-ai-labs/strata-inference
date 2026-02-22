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
        // Convert F16 to F32 at upload time — CPUs have no native F16 compute,
        // so all downstream ops (matmul, etc.) expect F32 data.
        if tensor.dtype() == TensorDtype::F16 {
            DeviceTensor::new(tensor.to_f32())
        } else {
            DeviceTensor::new(tensor.clone())
        }
    }

    fn download(&self, tensor: &DeviceTensor) -> Tensor {
        trace!(shape = ?tensor.shape(), dtype = ?tensor.dtype(), "CPU download (clone)");
        tensor.as_tensor().clone()
    }

    fn matmul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        // [M, K] x [K, N] -> [M, N]
        let a_data = a.as_tensor().as_f32();
        let b_data = b.as_tensor().as_f32();
        let a_shape = a.shape();
        let b_shape = b.shape();

        assert_eq!(
            a_shape.len(),
            2,
            "matmul: a must be 2D, got shape {:?}",
            a_shape
        );
        assert_eq!(
            b_shape.len(),
            2,
            "matmul: b must be 2D, got shape {:?}",
            b_shape
        );

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
        let a_data = a.as_tensor().as_f32();
        let b_data = b.as_tensor().as_f32();
        let a_shape = a.shape();
        let b_shape = b.shape();

        assert_eq!(
            a_shape.len(),
            2,
            "matmul_transpose: a must be 2D, got shape {:?}",
            a_shape
        );
        assert_eq!(
            b_shape.len(),
            2,
            "matmul_transpose: b must be 2D, got shape {:?}",
            b_shape
        );

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
        // Weights: [N, K] in quantized format
        // Input: [M, K] in F32
        // Output: [M, N] in F32
        // This is equivalent to input x weights^T but with fused dequantization.
        let input_data = input.as_tensor().as_f32();
        let input_shape = input.shape();
        let weight_shape = weights.shape();

        assert_eq!(input_shape.len(), 2, "quantized_matmul: input must be 2D");
        assert_eq!(
            weight_shape.len(),
            2,
            "quantized_matmul: weights must be 2D"
        );

        let dtype = weights.dtype();
        assert!(
            dtype.is_quantized(),
            "quantized_matmul: weights must be quantized, got {:?}",
            dtype
        );

        let m = input_shape[0];
        let k = input_shape[1];
        let n = weight_shape[0];

        assert_eq!(
            k, weight_shape[1],
            "quantized_matmul: input cols ({}) must match weight cols ({})",
            k, weight_shape[1]
        );

        let raw = match weights.as_tensor().storage() {
            TensorStorage::Quantized(data) => data,
            _ => panic!("quantized_matmul: expected Quantized storage"),
        };

        let block_size = dtype.block_size();
        let block_byte_size = dtype.block_byte_size();
        let blocks_per_row = (k + block_size - 1) / block_size;
        let bytes_per_row = blocks_per_row * block_byte_size;

        let mut result = vec![0.0f32; m * n];

        match dtype {
            TensorDtype::Q8_0 => {
                trace!(m, k, n, "CPU quantized_matmul (Q8_0)");

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

                            for e in elem_start..elem_end {
                                let qs = raw[block_start + 2 + (e - elem_start)] as i8;
                                sum += scale * (qs as f32) * input_row[e];
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q4_0 => {
                trace!(m, k, n, "CPU quantized_matmul (Q4_0)");

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let block_start = row_start + block_idx * 18;
                            let scale_bits =
                                u16::from_le_bytes([raw[block_start], raw[block_start + 1]]);
                            let scale = half::f16::from_bits(scale_bits).to_f32();

                            let elem_start = block_idx * 32;
                            let elem_end = std::cmp::min(elem_start + 32, k);

                            for e in elem_start..elem_end {
                                let local_idx = e - elem_start;
                                let nibble = if local_idx < 16 {
                                    raw[block_start + 2 + local_idx] & 0x0F
                                } else {
                                    (raw[block_start + 2 + local_idx - 16] >> 4) & 0x0F
                                };
                                let qs = nibble as i32 - 8;
                                sum += scale * (qs as f32) * input_row[e];
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q4_1 => {
                trace!(m, k, n, "CPU quantized_matmul (Q4_1)");

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let bs = row_start + block_idx * 20;
                            let d =
                                half::f16::from_bits(u16::from_le_bytes([raw[bs], raw[bs + 1]]))
                                    .to_f32();
                            let m_val = half::f16::from_bits(u16::from_le_bytes([
                                raw[bs + 2],
                                raw[bs + 3],
                            ]))
                            .to_f32();

                            let elem_start = block_idx * 32;
                            let elem_end = std::cmp::min(elem_start + 32, k);

                            for e in elem_start..elem_end {
                                let local_idx = e - elem_start;
                                let nibble = if local_idx < 16 {
                                    raw[bs + 4 + local_idx] & 0x0F
                                } else {
                                    (raw[bs + 4 + local_idx - 16] >> 4) & 0x0F
                                };
                                let dequant = nibble as f32 * d + m_val;
                                sum += dequant * input_row[e];
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q5_0 => {
                trace!(m, k, n, "CPU quantized_matmul (Q5_0)");

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let bs = row_start + block_idx * 22;
                            let d =
                                half::f16::from_bits(u16::from_le_bytes([raw[bs], raw[bs + 1]]))
                                    .to_f32();
                            let qh = u32::from_le_bytes([
                                raw[bs + 2],
                                raw[bs + 3],
                                raw[bs + 4],
                                raw[bs + 5],
                            ]);

                            let elem_start = block_idx * 32;
                            let elem_end = std::cmp::min(elem_start + 32, k);

                            for e in elem_start..elem_end {
                                let local_idx = e - elem_start;
                                let (nibble, high_bit) = if local_idx < 16 {
                                    let n = raw[bs + 6 + local_idx] & 0x0F;
                                    let h = ((qh >> (local_idx as u32)) << 4) & 0x10;
                                    (n as u32, h)
                                } else {
                                    let li = local_idx - 16;
                                    let n = (raw[bs + 6 + li] >> 4) & 0x0F;
                                    let h = (qh >> (li as u32 + 12)) & 0x10;
                                    (n as u32, h)
                                };
                                let qs = (nibble | high_bit) as i32 - 16;
                                sum += d * qs as f32 * input_row[e];
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q5_1 => {
                trace!(m, k, n, "CPU quantized_matmul (Q5_1)");

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let bs = row_start + block_idx * 24;
                            let d =
                                half::f16::from_bits(u16::from_le_bytes([raw[bs], raw[bs + 1]]))
                                    .to_f32();
                            let m_val = half::f16::from_bits(u16::from_le_bytes([
                                raw[bs + 2],
                                raw[bs + 3],
                            ]))
                            .to_f32();
                            let qh = u32::from_le_bytes([
                                raw[bs + 4],
                                raw[bs + 5],
                                raw[bs + 6],
                                raw[bs + 7],
                            ]);

                            let elem_start = block_idx * 32;
                            let elem_end = std::cmp::min(elem_start + 32, k);

                            for e in elem_start..elem_end {
                                let local_idx = e - elem_start;
                                let (nibble, high_bit) = if local_idx < 16 {
                                    let n = raw[bs + 8 + local_idx] & 0x0F;
                                    let h = ((qh >> (local_idx as u32)) << 4) & 0x10;
                                    (n as u32, h)
                                } else {
                                    let li = local_idx - 16;
                                    let n = (raw[bs + 8 + li] >> 4) & 0x0F;
                                    let h = (qh >> (li as u32 + 12)) & 0x10;
                                    (n as u32, h)
                                };
                                let val = (nibble | high_bit) as f32 * d + m_val;
                                sum += val * input_row[e];
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q4_K => {
                trace!(m, k, n, "CPU quantized_matmul (Q4_K)");
                use crate::gguf::quant::get_scale_min_k4;

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let bs = row_start + block_idx * 144;
                            let d_val =
                                half::f16::from_bits(u16::from_le_bytes([raw[bs], raw[bs + 1]]))
                                    .to_f32();
                            let dmin = half::f16::from_bits(u16::from_le_bytes([
                                raw[bs + 2],
                                raw[bs + 3],
                            ]))
                            .to_f32();
                            let scales: [u8; 12] = raw[bs + 4..bs + 16].try_into().unwrap();
                            let qs_start = bs + 16; // qs[128]

                            let elem_start = block_idx * 256;
                            let mut is = 0usize;
                            let mut q_off = 0usize;

                            for _chunk in 0..4 {
                                let (sc1, m1) = get_scale_min_k4(is, &scales);
                                let d1 = d_val * sc1 as f32;
                                let m1 = dmin * m1 as f32;
                                let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
                                let d2 = d_val * sc2 as f32;
                                let m2 = dmin * m2 as f32;

                                // First 32: low nibbles
                                for l in 0..32 {
                                    let e = elem_start + is / 2 * 64 + l;
                                    if e < k {
                                        let q = (raw[qs_start + q_off + l] & 0xF) as f32;
                                        sum += (d1 * q - m1) * input_row[e];
                                    }
                                }
                                // Next 32: high nibbles
                                for l in 0..32 {
                                    let e = elem_start + is / 2 * 64 + 32 + l;
                                    if e < k {
                                        let q = (raw[qs_start + q_off + l] >> 4) as f32;
                                        sum += (d2 * q - m2) * input_row[e];
                                    }
                                }
                                q_off += 32;
                                is += 2;
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q5_K => {
                trace!(m, k, n, "CPU quantized_matmul (Q5_K)");
                use crate::gguf::quant::get_scale_min_k4;

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let bs = row_start + block_idx * 176;
                            let d_val =
                                half::f16::from_bits(u16::from_le_bytes([raw[bs], raw[bs + 1]]))
                                    .to_f32();
                            let dmin = half::f16::from_bits(u16::from_le_bytes([
                                raw[bs + 2],
                                raw[bs + 3],
                            ]))
                            .to_f32();
                            let scales: [u8; 12] = raw[bs + 4..bs + 16].try_into().unwrap();
                            let qh_start = bs + 16; // qh[32]
                            let qs_start = bs + 48; // qs[128]

                            let elem_start = block_idx * 256;
                            let mut is = 0usize;
                            let mut ql_off = 0usize;
                            let mut u1: u8 = 1;
                            let mut u2: u8 = 2;

                            for _chunk in 0..4 {
                                let (sc1, m1) = get_scale_min_k4(is, &scales);
                                let d1 = d_val * sc1 as f32;
                                let m1 = dmin * m1 as f32;
                                let (sc2, m2) = get_scale_min_k4(is + 1, &scales);
                                let d2 = d_val * sc2 as f32;
                                let m2 = dmin * m2 as f32;

                                for l in 0..32 {
                                    let e = elem_start + is / 2 * 64 + l;
                                    if e < k {
                                        let high = if raw[qh_start + l] & u1 != 0 {
                                            16u32
                                        } else {
                                            0
                                        };
                                        let q = (raw[qs_start + ql_off + l] & 0xF) as u32 + high;
                                        sum += (d1 * q as f32 - m1) * input_row[e];
                                    }
                                }
                                for l in 0..32 {
                                    let e = elem_start + is / 2 * 64 + 32 + l;
                                    if e < k {
                                        let high = if raw[qh_start + l] & u2 != 0 {
                                            16u32
                                        } else {
                                            0
                                        };
                                        let q = (raw[qs_start + ql_off + l] >> 4) as u32 + high;
                                        sum += (d2 * q as f32 - m2) * input_row[e];
                                    }
                                }
                                ql_off += 32;
                                is += 2;
                                u1 = u1.wrapping_shl(2);
                                u2 = u2.wrapping_shl(2);
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            TensorDtype::Q6_K => {
                trace!(m, k, n, "CPU quantized_matmul (Q6_K)");

                for i in 0..m {
                    let input_row = &input_data[i * k..i * k + k];
                    for j in 0..n {
                        let row_start = j * bytes_per_row;
                        let mut sum = 0.0f32;

                        for block_idx in 0..blocks_per_row {
                            let bs = row_start + block_idx * 210;
                            // Q6_K layout: ql[128] | qh[64] | scales[16] | d(f16)
                            let ql_start = bs;
                            let qh_start = bs + 128;
                            let sc_start = bs + 192;
                            let d_val = half::f16::from_bits(u16::from_le_bytes([
                                raw[bs + 208],
                                raw[bs + 209],
                            ]))
                            .to_f32();

                            let elem_start = block_idx * 256;
                            let mut ql_off = 0usize;
                            let mut qh_off = 0usize;
                            let mut sc_off = 0usize;

                            for _n_chunk in 0..2 {
                                // 128 values per chunk
                                for l in 0..32 {
                                    let is = l / 16;
                                    let ql_byte0 = raw[ql_start + ql_off + l];
                                    let ql_byte1 = raw[ql_start + ql_off + l + 32];
                                    let qh_byte = raw[qh_start + qh_off + l];
                                    let sc = raw[sc_start + sc_off + is] as i8;

                                    let q1 = ((ql_byte0 & 0xF) | (((qh_byte >> 0) & 3) << 4))
                                        as i32
                                        - 32;
                                    let e1 = elem_start + _n_chunk * 128 + l;
                                    if e1 < k {
                                        sum += d_val * sc as f32 * q1 as f32 * input_row[e1];
                                    }

                                    let sc2 = raw[sc_start + sc_off + is + 2] as i8;
                                    let q2 = ((ql_byte1 & 0xF) | (((qh_byte >> 2) & 3) << 4))
                                        as i32
                                        - 32;
                                    let e2 = elem_start + _n_chunk * 128 + l + 32;
                                    if e2 < k {
                                        sum += d_val * sc2 as f32 * q2 as f32 * input_row[e2];
                                    }

                                    let sc3 = raw[sc_start + sc_off + is + 4] as i8;
                                    let q3 =
                                        ((ql_byte0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i32 - 32;
                                    let e3 = elem_start + _n_chunk * 128 + l + 64;
                                    if e3 < k {
                                        sum += d_val * sc3 as f32 * q3 as f32 * input_row[e3];
                                    }

                                    let sc4 = raw[sc_start + sc_off + is + 6] as i8;
                                    let q4 =
                                        ((ql_byte1 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i32 - 32;
                                    let e4 = elem_start + _n_chunk * 128 + l + 96;
                                    if e4 < k {
                                        sum += d_val * sc4 as f32 * q4 as f32 * input_row[e4];
                                    }
                                }
                                ql_off += 64;
                                qh_off += 32;
                                sc_off += 8;
                            }
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
            _ => unreachable!("quantized_matmul: unsupported dtype {:?}", dtype),
        }

        DeviceTensor::new(Tensor::new(vec![m, n], result))
    }

    fn add(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let a_data = a.as_tensor().as_f32();
        let b_data = b.as_tensor().as_f32();
        assert_eq!(
            a.shape(),
            b.shape(),
            "add: shapes must match, got {:?} and {:?}",
            a.shape(),
            b.shape()
        );

        trace!(shape = ?a.shape(), "CPU add");

        let result: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        DeviceTensor::new(Tensor::new(a.shape().to_vec(), result))
    }

    fn add_bias(&self, a: &DeviceTensor, bias: &DeviceTensor) -> DeviceTensor {
        // [M, N] + [N] -> [M, N]
        let a_data = a.as_tensor().as_f32();
        let bias_data = bias.as_tensor().as_f32();
        let a_shape = a.shape();
        let bias_shape = bias.shape();

        assert_eq!(
            a_shape.len(),
            2,
            "add_bias: a must be 2D, got shape {:?}",
            a_shape
        );
        assert_eq!(
            bias_shape.len(),
            1,
            "add_bias: bias must be 1D, got shape {:?}",
            bias_shape
        );

        let m = a_shape[0];
        let n = a_shape[1];
        assert_eq!(
            n, bias_shape[0],
            "add_bias: a cols ({}) must match bias length ({})",
            n, bias_shape[0]
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
        // Exact GELU: 0.5 * x * (1.0 + erf(x / sqrt(2)))
        // Matches llama.cpp's implementation using error function.
        let data = t.as_tensor().as_f32();
        trace!(n_elements = data.len(), "CPU gelu");

        let sqrt_2_inv: f32 = std::f32::consts::FRAC_1_SQRT_2; // 1/sqrt(2)

        let result: Vec<f32> = data
            .iter()
            .map(|&x| 0.5 * x * (1.0 + libm::erff(x * sqrt_2_inv)))
            .collect();

        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn silu(&self, t: &DeviceTensor) -> DeviceTensor {
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        let data = t.as_tensor().as_f32();
        trace!(n_elements = data.len(), "CPU silu");

        let result: Vec<f32> = data
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();

        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn swiglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        // SwiGLU: silu(gate) * up
        let gate_data = gate.as_tensor().as_f32();
        let up_data = up.as_tensor().as_f32();
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
        let data = t.as_tensor().as_f32();
        let w = weight.as_tensor().as_f32();
        let b = bias.as_tensor().as_f32();
        let shape = t.shape();

        // Works on the last dimension
        let ndim = shape.len();
        assert!(ndim >= 1, "layer_norm: tensor must be at least 1D");

        let hidden_size = shape[ndim - 1];
        let n_rows: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        assert_eq!(
            w.len(),
            hidden_size,
            "layer_norm: weight length must match hidden size"
        );
        assert_eq!(
            b.len(),
            hidden_size,
            "layer_norm: bias length must match hidden size"
        );

        trace!(n_rows, hidden_size, eps, "CPU layer_norm");

        let mut result = vec![0.0f32; data.len()];

        for row in 0..n_rows {
            let start = row * hidden_size;
            let end = start + hidden_size;
            let row_data = &data[start..end];

            // Compute mean
            let mean: f32 = row_data.iter().sum::<f32>() / hidden_size as f32;

            // Compute variance
            let variance: f32 = row_data
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f32>()
                / hidden_size as f32;

            let inv_std = 1.0 / (variance + eps).sqrt();

            for i in 0..hidden_size {
                result[start + i] = (row_data[i] - mean) * inv_std * w[i] + b[i];
            }
        }

        DeviceTensor::new(Tensor::new(shape.to_vec(), result))
    }

    fn rms_norm(&self, t: &DeviceTensor, weight: &DeviceTensor, eps: f32) -> DeviceTensor {
        // Per-row: x * rsqrt(mean(x^2) + eps) * weight
        let data = t.as_tensor().as_f32();
        let w = weight.as_tensor().as_f32();
        let shape = t.shape();

        let ndim = shape.len();
        assert!(ndim >= 1, "rms_norm: tensor must be at least 1D");

        let hidden_size = shape[ndim - 1];
        let n_rows: usize = shape[..ndim - 1].iter().product::<usize>().max(1);

        assert_eq!(
            w.len(),
            hidden_size,
            "rms_norm: weight length must match hidden size"
        );

        trace!(n_rows, hidden_size, eps, "CPU rms_norm");

        let mut result = vec![0.0f32; data.len()];

        for row in 0..n_rows {
            let start = row * hidden_size;
            let end = start + hidden_size;
            let row_data = &data[start..end];

            // mean(x^2)
            let mean_sq: f32 = row_data.iter().map(|&x| x * x).sum::<f32>() / hidden_size as f32;

            let rms_inv = 1.0 / (mean_sq + eps).sqrt();

            for i in 0..hidden_size {
                result[start + i] = row_data[i] * rms_inv * w[i];
            }
        }

        DeviceTensor::new(Tensor::new(shape.to_vec(), result))
    }

    fn softmax(&self, t: &DeviceTensor) -> DeviceTensor {
        // Per-row softmax (last dimension)
        let data = t.as_tensor().as_f32();
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
            let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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
        let data = t.as_tensor().as_f32();
        trace!(n_elements = data.len(), factor, "CPU scale");

        let result: Vec<f32> = data.iter().map(|&x| x * factor).collect();
        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn apply_causal_mask(&self, scores: &DeviceTensor, seq_len: usize) -> DeviceTensor {
        // For attention scores [seq_len, seq_len], set positions where col > row to -inf
        let data = scores.as_tensor().as_f32();
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
        rope_dim: usize,
    ) -> (DeviceTensor, DeviceTensor) {
        assert!(head_dim > 0, "rope: head_dim must be > 0");
        assert!(
            rope_dim <= head_dim,
            "rope: rope_dim ({}) must be <= head_dim ({})",
            rope_dim,
            head_dim
        );
        // q, k: [seq_len, total_dim] where total_dim = n_heads * head_dim
        // Apply rotary position embeddings to each head independently.
        // Only the first `rope_dim` dimensions of each head are rotated;
        // dimensions rope_dim..head_dim pass through unchanged.
        // For dimension pair (2i, 2i+1) within the rotated portion:
        //   freq[i] = 1.0 / (freq_base ^ (2i / rope_dim))
        //   cos_theta = cos(pos * freq[i])
        //   sin_theta = sin(pos * freq[i])
        //   q_rot[2i]   = q[2i]   * cos_theta - q[2i+1] * sin_theta
        //   q_rot[2i+1] = q[2i]   * sin_theta + q[2i+1] * cos_theta

        let q_data = q.as_tensor().as_f32();
        let k_data = k.as_tensor().as_f32();
        let q_shape = q.shape();
        let k_shape = k.shape();

        assert_eq!(
            q_shape.len(),
            2,
            "rope: q must be 2D, got shape {:?}",
            q_shape
        );
        assert_eq!(
            k_shape.len(),
            2,
            "rope: k must be 2D, got shape {:?}",
            k_shape
        );

        let seq_len = q_shape[0];
        let total_dim = q_shape[1];

        assert_eq!(k_shape[0], seq_len, "rope: q and k must have same seq_len");
        assert!(
            total_dim % head_dim == 0,
            "rope: total_dim ({}) must be divisible by head_dim ({})",
            total_dim,
            head_dim
        );
        assert!(
            rope_dim % 2 == 0,
            "rope: rope_dim ({}) must be even",
            rope_dim
        );
        assert!(
            rope_dim <= head_dim,
            "rope: rope_dim ({}) must be <= head_dim ({})",
            rope_dim,
            head_dim
        );

        let n_heads = total_dim / head_dim;
        let k_total_dim = k_shape[1];
        let k_n_heads = k_total_dim / head_dim;

        trace!(seq_len, total_dim, head_dim, rope_dim, n_heads, pos_offset, freq_base, "CPU rope");

        // Precompute frequencies for the rotated portion.
        // Matches llama.cpp: freq = pow(base, -1/rope_dim * 2*i)
        let half_rope_dim = rope_dim / 2;
        let inv_ndims = -1.0f32 / rope_dim as f32;
        let mut freqs = vec![0.0f32; half_rope_dim];
        for i in 0..half_rope_dim {
            freqs[i] = freq_base.powf(inv_ndims * (2 * i) as f32);
        }

        // Clone input data so non-rotated dims pass through unchanged
        let mut q_rot = q_data.to_vec();
        let mut k_rot = k_data.to_vec();

        for pos in 0..seq_len {
            let abs_pos = (pos + pos_offset) as f32;

            // Apply to Q
            for head in 0..n_heads {
                let offset = pos * total_dim + head * head_dim;
                for i in 0..half_rope_dim {
                    let theta = abs_pos * freqs[i];
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let q0 = q_data[offset + 2 * i];
                    let q1 = q_data[offset + 2 * i + 1];

                    q_rot[offset + 2 * i] = q0 * cos_theta - q1 * sin_theta;
                    q_rot[offset + 2 * i + 1] = q0 * sin_theta + q1 * cos_theta;
                }
                // dims rope_dim..head_dim are already correct from the clone
            }

            // Apply to K
            for head in 0..k_n_heads {
                let offset = pos * k_total_dim + head * head_dim;
                for i in 0..half_rope_dim {
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
        let data = hidden.as_tensor().as_f32();
        let shape = hidden.shape();

        assert_eq!(
            shape.len(),
            2,
            "mean_pool: hidden must be 2D, got shape {:?}",
            shape
        );

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
        let data = t.as_tensor().as_f32();
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

        trace!(
            vocab_size,
            hidden_size,
            n_tokens = ids.len(),
            "CPU embedding_lookup"
        );

        // Handle quantized embedding tables by dequantizing first
        let table_f32;
        let table_data = match table.as_tensor().dtype() {
            TensorDtype::F32 => table.as_tensor().as_f32(),
            _ => {
                table_f32 = table.as_tensor().to_f32();
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

    fn mul(&self, a: &DeviceTensor, b: &DeviceTensor) -> DeviceTensor {
        let a_data = a.as_tensor().as_f32();
        let b_data = b.as_tensor().as_f32();
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape == b_shape {
            // Same-shape element-wise multiply
            trace!(shape = ?a_shape, "CPU mul (same shape)");
            let result: Vec<f32> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(&x, &y)| x * y)
                .collect();
            DeviceTensor::new(Tensor::new(a_shape.to_vec(), result))
        } else if b_shape.len() == 1 && a_shape.len() == 2 && a_shape[1] == b_shape[0] {
            // Broadcast: [M, N] * [N] -> [M, N]
            let m = a_shape[0];
            let n = a_shape[1];
            trace!(m, n, "CPU mul (broadcast [M,N] * [N])");
            let mut result = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    result[i * n + j] = a_data[i * n + j] * b_data[j];
                }
            }
            DeviceTensor::new(Tensor::new(vec![m, n], result))
        } else {
            panic!("mul: unsupported shapes {:?} and {:?}", a_shape, b_shape);
        }
    }

    fn tanh(&self, t: &DeviceTensor) -> DeviceTensor {
        let data = t.as_tensor().as_f32();
        trace!(n_elements = data.len(), "CPU tanh");
        let result: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
        DeviceTensor::new(Tensor::new(t.shape().to_vec(), result))
    }

    fn geglu(&self, gate: &DeviceTensor, up: &DeviceTensor) -> DeviceTensor {
        let gate_data = gate.as_tensor().as_f32();
        let up_data = up.as_tensor().as_f32();
        assert_eq!(
            gate.shape(),
            up.shape(),
            "geglu: gate and up shapes must match, got {:?} and {:?}",
            gate.shape(),
            up.shape()
        );

        trace!(n_elements = gate_data.len(), "CPU geglu");

        let sqrt_2_over_pi: f32 = (2.0f32 / std::f32::consts::PI).sqrt();

        let result: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| {
                // GELU(gate) * up
                let inner = sqrt_2_over_pi * (g + 0.044715 * g * g * g);
                let gelu_g = 0.5 * g * (1.0 + inner.tanh());
                gelu_g * u
            })
            .collect();

        DeviceTensor::new(Tensor::new(gate.shape().to_vec(), result))
    }

    fn rope_neox(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        pos_offset: usize,
        freq_base: f32,
        head_dim: usize,
        rope_dim: usize,
    ) -> (DeviceTensor, DeviceTensor) {
        // NeoX-style RoPE: pairs (x[i], x[i + rope_dim/2]) instead of (x[2i], x[2i+1])
        assert!(head_dim > 0, "rope_neox: head_dim must be > 0");
        assert!(
            rope_dim <= head_dim,
            "rope_neox: rope_dim ({}) must be <= head_dim ({})",
            rope_dim,
            head_dim
        );
        assert!(
            rope_dim % 2 == 0,
            "rope_neox: rope_dim ({}) must be even",
            rope_dim
        );

        let q_data = q.as_tensor().as_f32();
        let k_data = k.as_tensor().as_f32();
        let q_shape = q.shape();
        let k_shape = k.shape();

        assert_eq!(
            q_shape.len(),
            2,
            "rope_neox: q must be 2D, got shape {:?}",
            q_shape
        );
        assert_eq!(
            k_shape.len(),
            2,
            "rope_neox: k must be 2D, got shape {:?}",
            k_shape
        );

        let seq_len = q_shape[0];
        let total_dim = q_shape[1];

        assert_eq!(
            k_shape[0], seq_len,
            "rope_neox: q and k must have same seq_len"
        );
        assert!(
            total_dim % head_dim == 0,
            "rope_neox: total_dim ({}) must be divisible by head_dim ({})",
            total_dim,
            head_dim
        );

        let n_heads = total_dim / head_dim;
        let k_total_dim = k_shape[1];
        let k_n_heads = k_total_dim / head_dim;

        trace!(
            seq_len,
            total_dim,
            head_dim,
            rope_dim,
            n_heads,
            pos_offset,
            freq_base,
            "CPU rope_neox"
        );

        // Matches llama.cpp: freq = pow(base, -1/rope_dim * 2*i)
        let half_rope_dim = rope_dim / 2;
        let inv_ndims = -1.0f32 / rope_dim as f32;
        let mut freqs = vec![0.0f32; half_rope_dim];
        for i in 0..half_rope_dim {
            freqs[i] = freq_base.powf(inv_ndims * (2 * i) as f32);
        }

        let mut q_rot = q_data.to_vec();
        let mut k_rot = k_data.to_vec();

        for pos in 0..seq_len {
            let abs_pos = (pos + pos_offset) as f32;

            // Apply to Q
            for head in 0..n_heads {
                let offset = pos * total_dim + head * head_dim;
                for i in 0..half_rope_dim {
                    let theta = abs_pos * freqs[i];
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    // NeoX pairing: (x[i], x[i + half_rope_dim])
                    let q0 = q_data[offset + i];
                    let q1 = q_data[offset + i + half_rope_dim];

                    q_rot[offset + i] = q0 * cos_theta - q1 * sin_theta;
                    q_rot[offset + i + half_rope_dim] = q0 * sin_theta + q1 * cos_theta;
                }
            }

            // Apply to K
            for head in 0..k_n_heads {
                let offset = pos * k_total_dim + head * head_dim;
                for i in 0..half_rope_dim {
                    let theta = abs_pos * freqs[i];
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let k0 = k_data[offset + i];
                    let k1 = k_data[offset + i + half_rope_dim];

                    k_rot[offset + i] = k0 * cos_theta - k1 * sin_theta;
                    k_rot[offset + i + half_rope_dim] = k0 * sin_theta + k1 * cos_theta;
                }
            }
        }

        (
            DeviceTensor::new(Tensor::new(q_shape.to_vec(), q_rot)),
            DeviceTensor::new(Tensor::new(k_shape.to_vec(), k_rot)),
        )
    }

    fn copy_rows_into(&self, dest: &DeviceTensor, src: &DeviceTensor, dest_row_offset: usize) {
        let dest_data = dest.as_tensor().as_f32();
        let src_data = src.as_tensor().as_f32();
        let cols = dest.shape().last().copied().unwrap_or(0);
        let n_rows = src.shape()[0];

        let mut result = dest_data.to_vec();
        let byte_offset = dest_row_offset * cols;
        let copy_len = n_rows * cols;
        result[byte_offset..byte_offset + copy_len].copy_from_slice(&src_data[..copy_len]);

        // Mutate in place by replacing storage (CPU tensors are cheap)
        // Since DeviceTensor is immutable, we can't truly mutate —
        // but KvCache will handle the indirection.
        // For CPU this is a no-op pattern; the caller keeps the new data.
        // Actually: this doesn't work with immutable DeviceTensor.
        // We need a different approach for CPU: just let KvCache hold the data.
        // This method is primarily useful for GPU backends.
        // For CPU, the KvCache will keep using Vec<f32> storage.
        panic!("copy_rows_into is only used with GPU KV cache; CPU uses Vec<f32> directly");
    }

    fn grouped_attention_decode(
        &self,
        q: &DeviceTensor,
        k: &DeviceTensor,
        v: &DeviceTensor,
        total_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_scale: f32,
        softcap: f32,
    ) -> DeviceTensor {
        let q_data = q.as_tensor().as_f32();
        let k_data = k.as_tensor().as_f32();
        let v_data = v.as_tensor().as_f32();
        let kv_dim = num_kv_heads * head_dim;
        let total_dim = num_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;

        let mut output = vec![0.0f32; total_dim];

        for h in 0..num_heads {
            let kv_head = h / heads_per_kv;
            let q_offset = h * head_dim;

            // Compute attention scores: dot(q_head, k[j, kv_head]) * scale
            let mut scores = vec![0.0f32; total_len];
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..total_len {
                let k_offset = j * kv_dim + kv_head * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_offset + d] * k_data[k_offset + d];
                }
                let mut score = dot * attn_scale;

                // Softcap
                if softcap > 0.0 {
                    score = softcap * (score / softcap).tanh();
                }

                scores[j] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for j in 0..total_len {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            }
            if sum_exp > 0.0 {
                let inv_sum = 1.0 / sum_exp;
                for j in 0..total_len {
                    scores[j] *= inv_sum;
                }
            }

            // Weighted sum of V
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..total_len {
                    let v_offset = j * kv_dim + kv_head * head_dim;
                    val += scores[j] * v_data[v_offset + d];
                }
                output[q_offset + d] = val;
            }
        }

        DeviceTensor::new(Tensor::new(vec![1, total_dim], output))
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
        assert_eq!(
            a.len(),
            b.len(),
            "{}: length mismatch {} vs {}",
            msg,
            a.len(),
            b.len()
        );
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
        assert_close(
            result.as_tensor().as_f32(),
            &[1.0, 2.0, 3.0, 4.0],
            1e-6,
            "matmul identity",
        );
    }

    #[test]
    fn test_matmul_basic() {
        let b = backend();
        // [2,3] x [3,2] -> [2,2]
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let bm = dt(Tensor::new(
            vec![3, 2],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ));
        let result = b.matmul(&a, &bm);
        assert_eq!(result.shape(), &[2, 2]);
        // Row 0: 1*7+2*9+3*11 = 7+18+33=58, 1*8+2*10+3*12=8+20+36=64
        // Row 1: 4*7+5*9+6*11=28+45+66=139, 4*8+5*10+6*12=32+50+72=154
        assert_close(
            result.as_tensor().as_f32(),
            &[58.0, 64.0, 139.0, 154.0],
            1e-5,
            "matmul basic",
        );
    }

    #[test]
    fn test_matmul_single_element() {
        let b = backend();
        let a = dt(Tensor::new(vec![1, 1], vec![3.0]));
        let bm = dt(Tensor::new(vec![1, 1], vec![4.0]));
        let result = b.matmul(&a, &bm);
        assert_close(result.as_tensor().as_f32(), &[12.0], 1e-6, "matmul single");
    }

    #[test]
    fn test_matmul_rectangular() {
        let b = backend();
        // [1,3] x [3,1] -> [1,1]
        let a = dt(Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]));
        let bm = dt(Tensor::new(vec![3, 1], vec![4.0, 5.0, 6.0]));
        let result = b.matmul(&a, &bm);
        assert_eq!(result.shape(), &[1, 1]);
        assert_close(result.as_tensor().as_f32(), &[32.0], 1e-5, "matmul rect");
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
        let bm = dt(Tensor::new(
            vec![2, 3],
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ));
        let result = b.matmul_transpose(&a, &bm);
        assert_eq!(result.shape(), &[2, 2]);
        assert_close(
            result.as_tensor().as_f32(),
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
        let bt = dt(Tensor::new(
            vec![2, 3],
            vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0],
        ));
        let result_transpose = b.matmul_transpose(&a, &bt);

        assert_close(
            result_normal.as_tensor().as_f32(),
            result_transpose.as_tensor().as_f32(),
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
            result_q.as_tensor().as_f32(),
            result_ref.as_tensor().as_f32(),
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
            result_q.as_tensor().as_f32(),
            result_ref.as_tensor().as_f32(),
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
        assert_close(
            result.as_tensor().as_f32(),
            &[11.0, 22.0, 33.0],
            1e-6,
            "add",
        );
    }

    #[test]
    fn test_add_2d() {
        let b = backend();
        let a = dt(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        let bv = dt(Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]));
        let result = b.add(&a, &bv);
        assert_close(
            result.as_tensor().as_f32(),
            &[11.0, 22.0, 33.0, 44.0],
            1e-6,
            "add 2d",
        );
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
            result.as_tensor().as_f32(),
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
        assert_close(result.as_tensor().as_f32(), &[0.0], 1e-6, "gelu(0)");
    }

    #[test]
    fn test_gelu_values() {
        let b = backend();
        // Known approximate GELU values:
        // gelu(1.0) ~ 0.8412
        // gelu(-1.0) ~ -0.1588
        let t = dt(Tensor::new(vec![2], vec![1.0, -1.0]));
        let result = b.gelu(&t);
        let data = result.as_tensor().as_f32();
        assert!((data[0] - 0.8412).abs() < 0.001, "gelu(1.0) = {}", data[0]);
        assert!(
            (data[1] - (-0.1588)).abs() < 0.001,
            "gelu(-1.0) = {}",
            data[1]
        );
    }

    #[test]
    fn test_gelu_large_positive() {
        let b = backend();
        // For large x, gelu(x) ~ x
        let t = dt(Tensor::new(vec![1], vec![10.0]));
        let result = b.gelu(&t);
        assert!(
            (result.as_tensor().as_f32()[0] - 10.0).abs() < 0.001,
            "gelu(10.0) ~ 10.0"
        );
    }

    // ---- silu ----

    #[test]
    fn test_silu_zero() {
        let b = backend();
        let t = dt(Tensor::new(vec![1], vec![0.0]));
        let result = b.silu(&t);
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert_close(result.as_tensor().as_f32(), &[0.0], 1e-6, "silu(0)");
    }

    #[test]
    fn test_silu_values() {
        let b = backend();
        // silu(1.0) = 1.0 * sigmoid(1.0) = 1.0 * 0.7311 = 0.7311
        // silu(-1.0) = -1.0 * sigmoid(-1.0) = -1.0 * 0.2689 = -0.2689
        let t = dt(Tensor::new(vec![2], vec![1.0, -1.0]));
        let result = b.silu(&t);
        let data = result.as_tensor().as_f32();
        assert!((data[0] - 0.7311).abs() < 0.001, "silu(1.0) = {}", data[0]);
        assert!(
            (data[1] - (-0.2689)).abs() < 0.001,
            "silu(-1.0) = {}",
            data[1]
        );
    }

    // ---- swiglu ----

    #[test]
    fn test_swiglu() {
        let b = backend();
        let gate = dt(Tensor::new(vec![3], vec![1.0, 0.0, -1.0]));
        let up = dt(Tensor::new(vec![3], vec![2.0, 3.0, 4.0]));
        let result = b.swiglu(&gate, &up);
        let data = result.as_tensor().as_f32();

        // swiglu = silu(gate) * up
        // silu(1.0) * 2.0 = 0.7311 * 2.0 = 1.4621
        // silu(0.0) * 3.0 = 0.0 * 3.0 = 0.0
        // silu(-1.0) * 4.0 = -0.2689 * 4.0 = -1.0756
        assert!((data[0] - 1.4621).abs() < 0.01, "swiglu[0] = {}", data[0]);
        assert!((data[1] - 0.0).abs() < 0.01, "swiglu[1] = {}", data[1]);
        assert!(
            (data[2] - (-1.0756)).abs() < 0.01,
            "swiglu[2] = {}",
            data[2]
        );
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

        let data = result.as_tensor().as_f32();
        // mean = 2.0, var = 1.0
        // (1 - 2) / sqrt(1 + 1e-5) = -1.0
        // (3 - 2) / sqrt(1 + 1e-5) = 1.0
        assert!(
            (data[0] - (-1.0)).abs() < 0.01,
            "layer_norm[0] = {}",
            data[0]
        );
        assert!((data[1] - 1.0).abs() < 0.01, "layer_norm[1] = {}", data[1]);
    }

    #[test]
    fn test_layer_norm_with_weight_bias() {
        let b = backend();
        let t = dt(Tensor::new(vec![1, 2], vec![1.0, 3.0]));
        let w = dt(Tensor::new(vec![2], vec![2.0, 2.0]));
        let bias = dt(Tensor::new(vec![2], vec![1.0, 1.0]));
        let result = b.layer_norm(&t, &w, &bias, 1e-5);
        let data = result.as_tensor().as_f32();
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
        let data = result.as_tensor().as_f32();

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
        let data = result.as_tensor().as_f32();

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
        let data = result.as_tensor().as_f32();

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
        let data = result.as_tensor().as_f32();

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
        let data = result.as_tensor().as_f32();
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
        let data = result.as_tensor().as_f32();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum = {}", sum);
    }

    #[test]
    fn test_softmax_multi_row() {
        let b = backend();
        let t = dt(Tensor::new(
            vec![2, 3],
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        ));
        let result = b.softmax(&t);
        let data = result.as_tensor().as_f32();

        // Each row should sum to 1
        let sum_row0: f32 = data[0..3].iter().sum();
        let sum_row1: f32 = data[3..6].iter().sum();
        assert!(
            (sum_row0 - 1.0).abs() < 1e-5,
            "softmax row0 sum = {}",
            sum_row0
        );
        assert!(
            (sum_row1 - 1.0).abs() < 1e-5,
            "softmax row1 sum = {}",
            sum_row1
        );
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let b = backend();
        // Large values should not overflow
        let t = dt(Tensor::new(vec![1, 3], vec![1000.0, 1001.0, 1002.0]));
        let result = b.softmax(&t);
        let data = result.as_tensor().as_f32();
        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax large values sum = {}",
            sum
        );
        assert!(
            data.iter().all(|&v| v.is_finite()),
            "softmax produced non-finite values"
        );
    }

    #[test]
    fn test_softmax_with_neg_infinity() {
        let b = backend();
        // Masked positions (neg infinity) should get zero probability
        let t = dt(Tensor::new(vec![1, 3], vec![1.0, f32::NEG_INFINITY, 2.0]));
        let result = b.softmax(&t);
        let data = result.as_tensor().as_f32();
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
        assert_close(result.as_tensor().as_f32(), &[2.5, 5.0, 7.5], 1e-6, "scale");
    }

    #[test]
    fn test_scale_zero() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
        let result = b.scale(&t, 0.0);
        assert_close(
            result.as_tensor().as_f32(),
            &[0.0, 0.0, 0.0],
            1e-6,
            "scale by 0",
        );
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
        let data = result.as_tensor().as_f32();

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
        assert_eq!(result.as_tensor().as_f32()[0], 5.0);
    }

    // ---- rope ----

    #[test]
    fn test_rope_zero_position() {
        let b = backend();
        // At position 0, cos(0)=1, sin(0)=0, so RoPE should be identity
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![5.0, 6.0, 7.0, 8.0]));
        let (q_rot, k_rot) = b.rope(&q, &k, 0, 10000.0, 4, 4);
        assert_close(
            q_rot.as_tensor().as_f32(),
            &[1.0, 2.0, 3.0, 4.0],
            1e-5,
            "rope q at pos 0",
        );
        assert_close(
            k_rot.as_tensor().as_f32(),
            &[5.0, 6.0, 7.0, 8.0],
            1e-5,
            "rope k at pos 0",
        );
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
        let (q_rot, _k_rot) = b.rope(&q, &k, 1, 1.0, 2, 2);

        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        let data = q_rot.as_tensor().as_f32();
        assert!((data[0] - cos1).abs() < 1e-5, "rope rot q[0] = {}", data[0]);
        assert!((data[1] - sin1).abs() < 1e-5, "rope rot q[1] = {}", data[1]);
    }

    #[test]
    fn test_rope_preserves_norm() {
        let b = backend();
        // RoPE is a rotation, so the L2 norm should be preserved
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![5.0, 6.0, 7.0, 8.0]));
        let (q_rot, _) = b.rope(&q, &k, 5, 10000.0, 4, 4);

        let norm_before: f32 = q
            .as_tensor()
            .as_f32()
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let norm_after: f32 = q_rot
            .as_tensor()
            .as_f32()
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
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
        let (q_rot, _) = b.rope(&q, &k, 1, 1.0, 2, 2);
        let data = q_rot.as_tensor().as_f32();

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
        let (q_rot, _) = b.rope(&q, &k, 0, 1.0, 2, 2);
        let data = q_rot.as_tensor().as_f32();

        // Pos 0: theta=0, cos=1, sin=0 -> [1, 0]
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        // Pos 1: theta=1, cos=cos(1), sin=sin(1) -> [cos1, sin1]
        let cos1 = 1.0f32.cos();
        let sin1 = 1.0f32.sin();
        assert!((data[2] - cos1).abs() < 1e-5);
        assert!((data[3] - sin1).abs() < 1e-5);
    }

    #[test]
    fn test_rope_partial_rotation() {
        let b = backend();
        // head_dim=8, rope_dim=4: first 4 dims rotated, last 4 unchanged
        // 1 head, seq_len=1
        let q = dt(Tensor::new(
            vec![1, 8],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ));
        let k = dt(Tensor::new(
            vec![1, 8],
            vec![1.0, 0.0, 1.0, 0.0, 9.0, 10.0, 11.0, 12.0],
        ));
        let (q_rot, k_rot) = b.rope(&q, &k, 5, 10000.0, 8, 4);

        let q_data = q_rot.as_tensor().as_f32();
        let k_data = k_rot.as_tensor().as_f32();

        // Last 4 dims (indices 4..8) should be unchanged
        assert!((q_data[4] - 5.0).abs() < 1e-6, "q[4] should be unchanged");
        assert!((q_data[5] - 6.0).abs() < 1e-6, "q[5] should be unchanged");
        assert!((q_data[6] - 7.0).abs() < 1e-6, "q[6] should be unchanged");
        assert!((q_data[7] - 8.0).abs() < 1e-6, "q[7] should be unchanged");

        assert!((k_data[4] - 9.0).abs() < 1e-6, "k[4] should be unchanged");
        assert!((k_data[5] - 10.0).abs() < 1e-6, "k[5] should be unchanged");
        assert!((k_data[6] - 11.0).abs() < 1e-6, "k[6] should be unchanged");
        assert!((k_data[7] - 12.0).abs() < 1e-6, "k[7] should be unchanged");

        // First 4 dims should be rotated (different from input at pos 5)
        let mut any_rotated = false;
        for i in 0..4 {
            if (q_data[i] - [1.0, 2.0, 3.0, 4.0][i]).abs() > 1e-6 {
                any_rotated = true;
                break;
            }
        }
        assert!(any_rotated, "First 4 dims of Q should be rotated at pos 5");

        // Norm of the rotated portion should be preserved (rotation is norm-preserving)
        let orig_norm: f32 = (1.0f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0).sqrt();
        let rot_norm: f32 = q_data[0..4].iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-4,
            "Partial RoPE should preserve norm: {} vs {}",
            orig_norm,
            rot_norm
        );
    }

    // ---- quantized_matmul Q4_0 ----

    #[test]
    fn test_quantized_matmul_q4_0() {
        let b = backend();

        // Create a 1x32 Q4_0 weight matrix
        // Q4_0 block: 2 bytes f16 scale + 16 bytes of nibbles (32 4-bit values)
        let scale = half::f16::from_f32(1.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_bits().to_le_bytes());
        // Pack 32 nibble values. Each byte holds 2 values:
        // Byte j's low nibble  → element j      (indices 0..15)
        // Byte j's high nibble → element j + 16  (indices 16..31)
        // We store byte=0x98 (low=8, high=9), so:
        //   elements 0..15: nibble=8, dequant=(8-8)*1.0=0
        //   elements 16..31: nibble=9, dequant=(9-8)*1.0=1
        for _ in 0..16 {
            raw.push(0x98); // low nibble=8, high nibble=9
        }

        let weights_q4 = Tensor::from_quantized(vec![1, 32], TensorDtype::Q4_0, raw);

        // Input: [1, 32] all ones
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let result = b.quantized_matmul(&dt(weights_q4), &dt(input));
        assert_eq!(result.shape(), &[1, 1]);

        // Expected: 16 zeros + 16 ones = 16.0
        let val = result.as_tensor().as_f32()[0];
        assert!(
            (val - 16.0).abs() < 0.5,
            "Q4_0 quantized_matmul: expected ~16.0, got {}",
            val
        );
    }

    #[test]
    fn test_quantized_matmul_q4_0_vs_reference() {
        let b = backend();

        // Create a 2x32 Q4_0 weight matrix with known pattern
        let scale = half::f16::from_f32(0.5);
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.extend_from_slice(&scale.to_bits().to_le_bytes());
            // All nibbles = 12 -> after -8 = 4, dequantized = 0.5 * 4 = 2.0
            for _ in 0..16 {
                raw.push(0xCC); // both nibbles = 12
            }
        }

        let weights_q4 = Tensor::from_quantized(vec![2, 32], TensorDtype::Q4_0, raw);

        // Input: [1, 32] all ones
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let result = b.quantized_matmul(&dt(weights_q4), &dt(input));
        assert_eq!(result.shape(), &[1, 2]);

        // Each row: 32 elements, each dequantized to 0.5 * 4 = 2.0
        // Dot with all-ones input: 32 * 2.0 = 64.0
        let data = result.as_tensor().as_f32();
        assert!(
            (data[0] - 64.0).abs() < 0.5,
            "Q4_0 row 0: expected ~64.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 64.0).abs() < 0.5,
            "Q4_0 row 1: expected ~64.0, got {}",
            data[1]
        );
    }

    #[test]
    fn test_quantized_matmul_q4_0_asymmetric_nibbles() {
        // Verify that quantized_matmul matches dequantize_q4_0 reference
        // when low and high nibbles differ within each byte.
        use crate::gguf::quant::{dequantize_q4_0, f32_to_f16, BlockQ4_0};

        let b = backend();
        let scale_f16 = f32_to_f16(1.0);

        // Byte pattern: low=15(0xF), high=0 → elements 0..15 get nibble 15,
        // elements 16..31 get nibble 0.
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = 0x0F; // low=15, high=0
        }

        let block = BlockQ4_0 { d: scale_f16, qs };
        let reference = dequantize_q4_0(&[block]);

        let mut raw = Vec::new();
        raw.extend_from_slice(&scale_f16.to_le_bytes());
        raw.extend_from_slice(&qs);

        let weights = Tensor::from_quantized(vec![1, 32], TensorDtype::Q4_0, raw);
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];

        // Reference: elements 0..15 = (15-8)*1.0 = 7.0, elements 16..31 = (0-8)*1.0 = -8.0
        // Sum = 16*7 + 16*(-8) = 112 - 128 = -16.0
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.5,
            "Q4_0 asymmetric matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    // ---- quantized_matmul for new K-quant types ----

    #[test]
    fn test_quantized_matmul_q4_1_vs_dequant() {
        use crate::gguf::quant::{dequantize_q4_1, f32_to_f16, BlockQ4_1};

        let b = backend();
        let block = BlockQ4_1 {
            d: f32_to_f16(0.5),
            m: f32_to_f16(0.25),
            qs: {
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    qs[j] = (j as u8) | ((j as u8) << 4);
                }
                qs
            },
        };
        let reference = dequantize_q4_1(&[block]);

        let mut raw = Vec::new();
        raw.extend_from_slice(&f32_to_f16(0.5).to_le_bytes());
        raw.extend_from_slice(&f32_to_f16(0.25).to_le_bytes());
        for j in 0..16u8 {
            raw.push(j | (j << 4));
        }

        let weights = Tensor::from_quantized(vec![1, 32], TensorDtype::Q4_1, raw);
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.1,
            "Q4_1 matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    #[test]
    fn test_quantized_matmul_q6_k_vs_dequant() {
        use crate::gguf::quant::{dequantize_q6_k, f32_to_f16, BlockQ6K};

        let b = backend();
        // Create a Q6_K block with known scale and data
        let mut block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: f32_to_f16(0.1),
        };
        block.scales[0] = 2; // First 16 elements use scale 2

        let reference = dequantize_q6_k(&[block]);

        // Build raw bytes in block layout order: ql[128] | qh[64] | scales[16] | d(2)
        let mut raw = Vec::new();
        raw.extend_from_slice(&block.ql);
        raw.extend_from_slice(&block.qh);
        raw.extend_from_slice(unsafe { &std::mem::transmute::<[i8; 16], [u8; 16]>(block.scales) });
        raw.extend_from_slice(&block.d.to_le_bytes());
        assert_eq!(raw.len(), 210);

        let weights = Tensor::from_quantized(vec![1, 256], TensorDtype::Q6_K, raw);
        let input = Tensor::new(vec![1, 256], vec![1.0f32; 256]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.1,
            "Q6_K matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    #[test]
    fn test_quantized_matmul_q4_k_vs_dequant() {
        use crate::gguf::quant::{dequantize_q4_k, f32_to_f16, BlockQ4K};

        let b = backend();
        let mut block = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [0; 12],
            qs: [0x55; 128], // low=5, high=5
        };
        block.scales[0] = 1; // Scale for sub-block 0
        block.scales[1] = 2; // Scale for sub-block 1
        block.scales[4] = 1; // Min for sub-block 0
        block.scales[5] = 1; // Min for sub-block 1

        let reference = dequantize_q4_k(&[block]);

        // Build raw bytes
        let mut raw = Vec::new();
        raw.extend_from_slice(&block.d.to_le_bytes());
        raw.extend_from_slice(&block.dmin.to_le_bytes());
        raw.extend_from_slice(&block.scales);
        raw.extend_from_slice(&block.qs);
        assert_eq!(raw.len(), 144);

        let weights = Tensor::from_quantized(vec![1, 256], TensorDtype::Q4_K, raw);
        let input = Tensor::new(vec![1, 256], vec![1.0f32; 256]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.1,
            "Q4_K matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    #[test]
    fn test_quantized_matmul_q5_0_vs_dequant() {
        use crate::gguf::quant::{dequantize_q5_0, f32_to_f16, BlockQ5_0};

        let b = backend();
        // d=2.0, qs with varied nibbles, qh with some bits set
        let block = BlockQ5_0 {
            d: f32_to_f16(2.0),
            qh: {
                // Set alternating bits: 0x55555555 = bits 0,2,4,...
                let val: u32 = 0x55555555;
                val.to_le_bytes()
            },
            qs: {
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    qs[j] = ((j as u8) & 0xF) | (((15 - j) as u8) << 4);
                }
                qs
            },
        };
        let reference = dequantize_q5_0(&[block]);

        // Build raw bytes: d(2) | qh(4) | qs(16) = 22 bytes
        let mut raw = Vec::new();
        raw.extend_from_slice(&block.d.to_le_bytes());
        raw.extend_from_slice(&block.qh);
        raw.extend_from_slice(&block.qs);
        assert_eq!(raw.len(), 22);

        let weights = Tensor::from_quantized(vec![1, 32], TensorDtype::Q5_0, raw);
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.1,
            "Q5_0 matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    #[test]
    fn test_quantized_matmul_q5_1_vs_dequant() {
        use crate::gguf::quant::{dequantize_q5_1, f32_to_f16, BlockQ5_1};

        let b = backend();
        let block = BlockQ5_1 {
            d: f32_to_f16(0.5),
            m: f32_to_f16(1.0),
            qh: {
                let val: u32 = 0xAAAAAAAA; // bits 1,3,5,...
                val.to_le_bytes()
            },
            qs: {
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    qs[j] = 0xA5;
                } // low=5, high=10
                qs
            },
        };
        let reference = dequantize_q5_1(&[block]);

        // Build raw bytes: d(2) | m(2) | qh(4) | qs(16) = 24 bytes
        let mut raw = Vec::new();
        raw.extend_from_slice(&block.d.to_le_bytes());
        raw.extend_from_slice(&block.m.to_le_bytes());
        raw.extend_from_slice(&block.qh);
        raw.extend_from_slice(&block.qs);
        assert_eq!(raw.len(), 24);

        let weights = Tensor::from_quantized(vec![1, 32], TensorDtype::Q5_1, raw);
        let input = Tensor::new(vec![1, 32], vec![1.0f32; 32]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.1,
            "Q5_1 matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    #[test]
    fn test_quantized_matmul_q5_k_vs_dequant() {
        use crate::gguf::quant::{dequantize_q5_k, f32_to_f16, BlockQ5K};

        let b = backend();
        let mut block = BlockQ5K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.5),
            scales: [0; 12],
            qh: [0xFF; 32],  // all high bits set — exercises wrapping_shl path
            qs: [0x33; 128], // nibbles = 3
        };
        // Set scales for first 4 sub-blocks
        for i in 0..4 {
            block.scales[i] = 2; // scale
            block.scales[i + 4] = 1; // min
        }

        let reference = dequantize_q5_k(&[block]);

        // Build raw bytes: d(2) | dmin(2) | scales(12) | qh(32) | qs(128) = 176
        let mut raw = Vec::new();
        raw.extend_from_slice(&block.d.to_le_bytes());
        raw.extend_from_slice(&block.dmin.to_le_bytes());
        raw.extend_from_slice(&block.scales);
        raw.extend_from_slice(&block.qh);
        raw.extend_from_slice(&block.qs);
        assert_eq!(raw.len(), 176);

        let weights = Tensor::from_quantized(vec![1, 256], TensorDtype::Q5_K, raw);
        let input = Tensor::new(vec![1, 256], vec![1.0f32; 256]);

        let result = b.quantized_matmul(&dt(weights), &dt(input));
        let matmul_val = result.as_tensor().as_f32()[0];
        let ref_sum: f32 = reference.iter().sum();
        assert!(
            (matmul_val - ref_sum).abs() < 0.5,
            "Q5_K matmul ({}) must match dequant reference ({})",
            matmul_val,
            ref_sum
        );
    }

    // ---- mean_pool ----

    #[test]
    fn test_mean_pool_all_ones_mask() {
        let b = backend();
        let hidden = dt(Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let mask = vec![1.0, 1.0, 1.0];
        let result = b.mean_pool(&hidden, &mask);
        assert_eq!(result.shape(), &[2]);
        // Mean: [(1+3+5)/3, (2+4+6)/3] = [3, 4]
        assert_close(
            result.as_tensor().as_f32(),
            &[3.0, 4.0],
            1e-5,
            "mean_pool all ones",
        );
    }

    #[test]
    fn test_mean_pool_partial_mask() {
        let b = backend();
        let hidden = dt(Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let mask = vec![1.0, 1.0, 0.0]; // Only first two tokens
        let result = b.mean_pool(&hidden, &mask);
        // Mean: [(1+3)/2, (2+4)/2] = [2, 3]
        assert_close(
            result.as_tensor().as_f32(),
            &[2.0, 3.0],
            1e-5,
            "mean_pool partial mask",
        );
    }

    #[test]
    fn test_mean_pool_zero_mask() {
        let b = backend();
        let hidden = dt(Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        let mask = vec![0.0, 0.0];
        let result = b.mean_pool(&hidden, &mask);
        assert_close(
            result.as_tensor().as_f32(),
            &[0.0, 0.0],
            1e-5,
            "mean_pool zero mask",
        );
    }

    // ---- l2_normalize ----

    #[test]
    fn test_l2_normalize() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![3.0, 4.0, 0.0]));
        let result = b.l2_normalize(&t);
        let data = result.as_tensor().as_f32();
        // norm = 5
        assert_close(data, &[0.6, 0.8, 0.0], 1e-5, "l2_normalize");
    }

    #[test]
    fn test_l2_normalize_unit_vector() {
        let b = backend();
        let t = dt(Tensor::new(vec![2], vec![0.6, 0.8]));
        let result = b.l2_normalize(&t);
        let data = result.as_tensor().as_f32();
        let norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized vector should have norm 1, got {}",
            norm
        );
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![0.0, 0.0, 0.0]));
        let result = b.l2_normalize(&t);
        assert_close(
            result.as_tensor().as_f32(),
            &[0.0, 0.0, 0.0],
            1e-6,
            "l2_normalize zero",
        );
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
            result.as_tensor().as_f32(),
            &[2.1, 2.2, 2.3, 0.1, 0.2, 0.3, 3.1, 3.2, 3.3],
            1e-6,
            "embedding_lookup",
        );
    }

    #[test]
    fn test_embedding_lookup_single() {
        let b = backend();
        let table = dt(Tensor::new(vec![3, 2], vec![0.0, 0.1, 1.0, 1.1, 2.0, 2.1]));
        let result = b.embedding_lookup(&table, &[1]);
        assert_eq!(result.shape(), &[1, 2]);
        assert_close(
            result.as_tensor().as_f32(),
            &[1.0, 1.1],
            1e-6,
            "embedding single",
        );
    }

    #[test]
    fn test_embedding_lookup_repeated() {
        let b = backend();
        let table = dt(Tensor::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]));
        let result = b.embedding_lookup(&table, &[0, 0, 1, 0]);
        assert_eq!(result.shape(), &[4, 2]);
        assert_close(
            result.as_tensor().as_f32(),
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
        let data = probs.as_tensor().as_f32();

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
            result.as_tensor().as_f32(),
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
        let norm_data = normed.as_tensor().as_f32();
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
        let data = result.as_tensor().as_f32();

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
        let data = result.as_tensor().as_f32();

        // gelu(-5) should be close to 0
        assert!(
            data[0].abs() < 0.01,
            "gelu(-5) should be ~0, got {}",
            data[0]
        );
        // gelu(0) = 0
        assert!(
            (data[2]).abs() < 1e-6,
            "gelu(0) should be 0, got {}",
            data[2]
        );
        // gelu(5) should be close to 5
        assert!(
            (data[4] - 5.0).abs() < 0.01,
            "gelu(5) should be ~5, got {}",
            data[4]
        );
        // For positive inputs, gelu should be monotonically increasing
        assert!(
            data[3] < data[4],
            "gelu should increase for positive values"
        );
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
        let data = result.as_tensor().as_f32();
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
        let norm_data = normed.as_tensor().as_f32().to_vec();

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
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ));
        let gate = b.matmul(&normed, &gate_weights);
        let up = b.matmul(&normed, &gate_weights);

        // Step 3: SwiGLU
        let swiglu_result = b.swiglu(&gate, &up);
        let swiglu_data = swiglu_result.as_tensor().as_f32();

        // Verify: swiglu(x, x) = silu(x) * x
        for i in 0..4 {
            let x = norm_data[i];
            let silu_x = x / (1.0 + (-x).exp());
            let expected = silu_x * x;
            assert!(
                (swiglu_data[i] - expected).abs() < 1e-4,
                "swiglu at {}: got {}, expected {}",
                i,
                swiglu_data[i],
                expected
            );
        }

        // Step 4: Add residual (original x)
        let residual = b.add(&swiglu_result, &x);
        assert_eq!(residual.shape(), &[1, 4]);
        // Just verify it runs without error and produces finite values
        for &v in residual.as_tensor().as_f32() {
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
        assert_close(
            result.as_tensor().as_f32(),
            &[12.0],
            1e-6,
            "matmul_transpose 1x1",
        );
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
        let data = result.as_tensor().as_f32();
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
        let data = result.as_tensor().as_f32();
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
        assert!(
            (result.as_tensor().as_f32()[0] - 1.0).abs() < 1e-6,
            "softmax of single element"
        );
    }

    // -- Scale with negative factor --

    #[test]
    fn test_scale_negative() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![1.0, -2.0, 3.0]));
        let result = b.scale(&t, -1.0);
        assert_close(
            result.as_tensor().as_f32(),
            &[-1.0, 2.0, -3.0],
            1e-6,
            "scale by -1",
        );
    }

    // -- Causal mask on non-square matrix --

    #[test]
    fn test_causal_mask_2x3() {
        let b = backend();
        let scores = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let result = b.apply_causal_mask(&scores, 2);
        let data = result.as_tensor().as_f32();
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
        let (q_rot, k_rot) = b.rope(&q, &k, 3, 10000.0, 8, 8);

        // Verify output shapes
        assert_eq!(q_rot.shape(), &[1, 16]);
        assert_eq!(k_rot.shape(), &[1, 16]);

        // RoPE is a rotation, so norms should be preserved per head.
        // Head 0: elements 0..8
        let q_norm_before: f32 = (0..8).map(|i| (i as f32 * 0.1).powi(2)).sum::<f32>().sqrt();
        let q_norm_after: f32 = q_rot.as_tensor().as_f32()[0..8]
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
        let (q_rot, k_rot) = b.rope(&q, &k, 1, 10000.0, 2, 2);
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
        assert_close(
            result.as_tensor().as_f32(),
            &[1.0, 2.0, 3.0],
            1e-5,
            "single token mean_pool",
        );
    }

    // -- L2 normalize already-unit vector --

    #[test]
    fn test_l2_normalize_already_normalized() {
        let b = backend();
        // Already unit norm
        let t = dt(Tensor::new(vec![2], vec![0.6, 0.8]));
        let result = b.l2_normalize(&t);
        let data = result.as_tensor().as_f32();
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
        assert!((result.as_tensor().as_f32()[0] - 1.0).abs() < 1e-6);
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
        let table = DeviceTensor::new(Tensor::from_quantized(vec![3, 32], TensorDtype::Q8_0, raw));
        let result = b.embedding_lookup(&table, &[1, 0]);
        assert_eq!(result.shape(), &[2, 32]);
        // Verify it returns finite values
        for &v in result.as_tensor().as_f32() {
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
        assert!(result.as_tensor().as_f32().is_empty());
    }

    // -- Silu symmetry --

    #[test]
    fn test_silu_symmetry() {
        let b = backend();
        // silu(-x) = -x * sigmoid(-x) = -x * (1 - sigmoid(x))
        // silu(x) + silu(-x) != 0 (it's not odd), but let's verify the math.
        let t_pos = dt(Tensor::new(vec![1], vec![2.0]));
        let t_neg = dt(Tensor::new(vec![1], vec![-2.0]));
        let pos = b.silu(&t_pos).as_tensor().as_f32()[0];
        let neg = b.silu(&t_neg).as_tensor().as_f32()[0];
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
        assert_close(
            result.as_tensor().as_f32(),
            &[0.0, 0.0, 0.0],
            1e-6,
            "swiglu zero gate",
        );
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
        let data = output.as_tensor().as_f32();
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
        for &v in activated.as_tensor().as_f32() {
            assert!(v.is_finite(), "layer_norm->gelu produced non-finite");
        }
    }

    // -- Embedding lookup -> layer_norm -> mean_pool pipeline --

    #[test]
    fn test_embed_norm_pool_pipeline() {
        let b = backend();
        let table = dt(Tensor::new(
            vec![4, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
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
        let norm: f32 = normalized
            .as_tensor()
            .as_f32()
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "l2 norm should be 1.0, got {}",
            norm
        );
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
        let data = result.as_tensor().as_f32();
        for i in 0..rows {
            for j in 0..cols {
                let expected = a_data[i * cols + j] + bias_data[j];
                assert!(
                    (data[i * cols + j] - expected).abs() < 1e-6,
                    "add_bias[{},{}]: got {}, expected {}",
                    i,
                    j,
                    data[i * cols + j],
                    expected
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
        let val = result.as_tensor().as_f32()[0];
        assert!(
            (val - 32.0).abs() < 0.5,
            "identity-like Q8_0 matmul: got {}",
            val
        );
    }

    // --- Tests for Phase 0 new methods ---

    #[test]
    fn test_mul_same_shape() {
        let b = backend();
        let a = dt(Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
        let c = dt(Tensor::new(vec![4], vec![5.0, 6.0, 7.0, 8.0]));
        let result = b.mul(&a, &c);
        assert_eq!(result.as_tensor().as_f32(), &[5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_mul_broadcast() {
        let b = backend();
        let a = dt(Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let c = dt(Tensor::new(vec![3], vec![10.0, 20.0, 30.0]));
        let result = b.mul(&a, &c);
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_tensor().as_f32(),
            &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0]
        );
    }

    #[test]
    fn test_mul_zeros() {
        let b = backend();
        let a = dt(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
        let c = dt(Tensor::new(vec![3], vec![0.0, 0.0, 0.0]));
        let result = b.mul(&a, &c);
        assert_eq!(result.as_tensor().as_f32(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tanh_basic() {
        let b = backend();
        let t = dt(Tensor::new(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]));
        let result = b.tanh(&t);
        let data = result.as_tensor().as_f32();
        assert!((data[0] - (-2.0f32).tanh()).abs() < 1e-6);
        assert!((data[1] - (-1.0f32).tanh()).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0f32.tanh()).abs() < 1e-6);
        assert!((data[4] - 2.0f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_bounds() {
        let b = backend();
        let t = dt(Tensor::new(vec![3], vec![-100.0, 0.0, 100.0]));
        let result = b.tanh(&t);
        let data = result.as_tensor().as_f32();
        assert!((data[0] - (-1.0)).abs() < 1e-6, "tanh(-100) should be ~-1");
        assert!((data[1] - 0.0).abs() < 1e-6, "tanh(0) should be 0");
        assert!((data[2] - 1.0).abs() < 1e-6, "tanh(100) should be ~1");
    }

    #[test]
    fn test_geglu_basic() {
        let b = backend();
        let gate = dt(Tensor::new(vec![4], vec![-1.0, 0.0, 1.0, 2.0]));
        let up = dt(Tensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
        let result = b.geglu(&gate, &up);
        let data = result.as_tensor().as_f32();
        assert_eq!(result.shape(), &[4]);

        // geglu(-1, 1) = gelu(-1) * 1; gelu(-1) ≈ -0.1588
        assert!(
            (data[0] - (-0.1588)).abs() < 0.01,
            "geglu(-1, 1) = {}",
            data[0]
        );
        // geglu(0, 2) = gelu(0) * 2 = 0
        assert!((data[1] - 0.0).abs() < 1e-6, "geglu(0, 2) = {}", data[1]);
        // geglu(1, 3) = gelu(1) * 3; gelu(1) ≈ 0.8412
        assert!(
            (data[2] - 0.8412 * 3.0).abs() < 0.05,
            "geglu(1, 3) = {}",
            data[2]
        );
        // geglu(2, 4) = gelu(2) * 4; gelu(2) ≈ 1.9545
        assert!(
            (data[3] - 1.9545 * 4.0).abs() < 0.05,
            "geglu(2, 4) = {}",
            data[3]
        );
    }

    #[test]
    fn test_geglu_zero_up() {
        let b = backend();
        let gate = dt(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
        let up = dt(Tensor::new(vec![3], vec![0.0, 0.0, 0.0]));
        let result = b.geglu(&gate, &up);
        let data = result.as_tensor().as_f32();
        for &v in data {
            assert!((v - 0.0).abs() < 1e-6, "geglu(*, 0) should be 0, got {}", v);
        }
    }

    #[test]
    fn test_rope_neox_basic() {
        let b = backend();
        // seq_len=1, 1 head, head_dim=4, rope_dim=4
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 0.0, 0.0, 0.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![0.0, 1.0, 0.0, 0.0]));
        let (q_rot, k_rot) = b.rope_neox(&q, &k, 0, 10000.0, 4, 4);
        // At position 0, all rotations are by angle 0, so cos=1, sin=0 -> no change
        let q_data = q_rot.as_tensor().as_f32();
        let k_data = k_rot.as_tensor().as_f32();
        assert!(
            (q_data[0] - 1.0).abs() < 1e-5,
            "rope_neox pos=0 should preserve q"
        );
        assert!(
            (k_data[1] - 1.0).abs() < 1e-5,
            "rope_neox pos=0 should preserve k"
        );
    }

    #[test]
    fn test_rope_neox_preserves_norm() {
        let b = backend();
        // seq_len=2, 2 heads, head_dim=4, rope_dim=4
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let k_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.5).collect();
        let q = dt(Tensor::new(vec![2, 8], q_data.clone()));
        let k = dt(Tensor::new(vec![2, 8], k_data.clone()));

        let (q_rot, k_rot) = b.rope_neox(&q, &k, 3, 10000.0, 4, 4);
        let q_rot_data = q_rot.as_tensor().as_f32();
        let k_rot_data = k_rot.as_tensor().as_f32();

        // RoPE is a rotation, so it should preserve the norm of each head
        for head_start in (0..16).step_by(4) {
            let orig_q_norm: f32 = (0..4)
                .map(|j| q_data[head_start + j].powi(2))
                .sum::<f32>()
                .sqrt();
            let rot_q_norm: f32 = (0..4)
                .map(|j| q_rot_data[head_start + j].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                (orig_q_norm - rot_q_norm).abs() < 1e-4,
                "rope_neox Q norm: orig={}, rot={}",
                orig_q_norm,
                rot_q_norm
            );

            let orig_k_norm: f32 = (0..4)
                .map(|j| k_data[head_start + j].powi(2))
                .sum::<f32>()
                .sqrt();
            let rot_k_norm: f32 = (0..4)
                .map(|j| k_rot_data[head_start + j].powi(2))
                .sum::<f32>()
                .sqrt();
            assert!(
                (orig_k_norm - rot_k_norm).abs() < 1e-4,
                "rope_neox K norm: orig={}, rot={}",
                orig_k_norm,
                rot_k_norm
            );
        }
    }

    #[test]
    fn test_rope_neox_differs_from_rope() {
        let b = backend();
        // Verify that rope_neox produces different results than rope for the same input
        // seq_len=1, 1 head, head_dim=4, rope_dim=4
        let q_data = vec![1.0, 2.0, 3.0, 4.0];
        let k_data = vec![5.0, 6.0, 7.0, 8.0];
        let q = dt(Tensor::new(vec![1, 4], q_data));
        let k = dt(Tensor::new(vec![1, 4], k_data));

        // pos_offset=5 so there's actual rotation (not just pos=0)
        let (q_norm, _) = b.rope(&q, &k, 5, 10000.0, 4, 4);
        let (q_neox, _) = b.rope_neox(&q, &k, 5, 10000.0, 4, 4);

        let q_norm_data = q_norm.as_tensor().as_f32();
        let q_neox_data = q_neox.as_tensor().as_f32();

        // They should differ (different pairing schemes)
        let differs = q_norm_data
            .iter()
            .zip(q_neox_data)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "rope_neox should produce different results from rope for non-zero positions"
        );
    }

    #[test]
    fn test_rope_neox_partial_rotation() {
        let b = backend();
        // head_dim=8 but only rotate first 4 dims
        let q_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k_data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let q = dt(Tensor::new(vec![1, 8], q_data));
        let k = dt(Tensor::new(vec![1, 8], k_data));

        let (q_rot, k_rot) = b.rope_neox(&q, &k, 5, 10000.0, 8, 4);
        let q_rot_data = q_rot.as_tensor().as_f32();
        let k_rot_data = k_rot.as_tensor().as_f32();

        // Dims 4-7 should be unchanged (pass-through)
        assert!(
            (q_rot_data[4] - 5.0).abs() < 1e-6,
            "rope_neox should pass through unrotated dims"
        );
        assert!((q_rot_data[5] - 6.0).abs() < 1e-6);
        assert!((q_rot_data[6] - 7.0).abs() < 1e-6);
        assert!((q_rot_data[7] - 8.0).abs() < 1e-6);
        assert!((k_rot_data[4] - 1.0).abs() < 1e-6);
        assert!((k_rot_data[5] - 1.0).abs() < 1e-6);
        assert!((k_rot_data[6] - 1.0).abs() < 1e-6);
        assert!((k_rot_data[7] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_select_backend() {
        let backend = crate::backend::select_backend();
        // Should always return a valid backend (at minimum CPU)
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let dt = backend.upload(&t);
        let result = backend.download(&dt);
        assert_eq!(result.as_f32(), &[1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // grouped_attention_decode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grouped_attention_decode_basic() {
        // Simple MHA: 2 heads, 2 KV heads, head_dim=2, total_len=3
        let b = backend();
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 2;
        let total_len = 3;

        // Q: [1, 4] (2 heads * 2 dim)
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 0.0, 0.0, 1.0]));
        // K: [3, 4], V: [3, 4]
        let k = dt(Tensor::new(
            vec![3, 4],
            vec![
                1.0, 0.0, 0.0, 1.0, // pos 0
                0.0, 1.0, 1.0, 0.0, // pos 1
                1.0, 1.0, 1.0, 1.0, // pos 2
            ],
        ));
        let v = dt(Tensor::new(
            vec![3, 4],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        ));

        let result = b.grouped_attention_decode(
            &q,
            &k,
            &v,
            total_len,
            num_heads,
            num_kv_heads,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            0.0,
        );

        assert_eq!(result.shape(), &[1, 4]);
        let out = b.download(&result).as_f32().to_vec();
        // Just verify shape and finite values
        assert_eq!(out.len(), 4);
        for &val in &out {
            assert!(val.is_finite(), "output contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_grouped_attention_decode_gqa() {
        // GQA: 4 Q heads, 2 KV heads, head_dim=2, total_len=2
        let b = backend();
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 2;
        let total_len = 2;

        // Q: [1, 8] (4 heads * 2 dim)
        let q = dt(Tensor::new(
            vec![1, 8],
            vec![
                1.0, 0.0, // head 0 (maps to kv_head 0)
                0.0, 1.0, // head 1 (maps to kv_head 0)
                1.0, 1.0, // head 2 (maps to kv_head 1)
                -1.0, 0.0, // head 3 (maps to kv_head 1)
            ],
        ));
        // K: [2, 4] (2 kv_heads * 2 dim)
        let k = dt(Tensor::new(
            vec![2, 4],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        ));
        // V: [2, 4]
        let v = dt(Tensor::new(
            vec![2, 4],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ));

        let result = b.grouped_attention_decode(
            &q,
            &k,
            &v,
            total_len,
            num_heads,
            num_kv_heads,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
            0.0,
        );

        assert_eq!(result.shape(), &[1, 8]);
        let out = b.download(&result).as_f32().to_vec();
        assert_eq!(out.len(), 8);
        for &val in &out {
            assert!(val.is_finite(), "output contains non-finite value: {}", val);
        }

        // Heads 0 and 1 share kv_head 0, heads 2 and 3 share kv_head 1
        // Probabilities should sum to 1 per head (softmax property)
        // Head 0 (q=[1,0]) with K kv_head 0: scores are [1*1+0*0, 1*0+0*1]=[1,0]
        // After softmax with scale 1/sqrt(2): should attend more to pos 0
    }

    #[test]
    fn test_grouped_attention_decode_softcap() {
        // Test softcap: should limit attention scores
        let b = backend();
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 2;
        let total_len = 2;

        // Large Q to produce large dot products
        let q = dt(Tensor::new(vec![1, 2], vec![10.0, 10.0]));
        let k = dt(Tensor::new(vec![2, 2], vec![10.0, 10.0, -10.0, -10.0]));
        let v = dt(Tensor::new(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]));

        // Without softcap
        let result_no_cap = b.grouped_attention_decode(
            &q,
            &k,
            &v,
            total_len,
            num_heads,
            num_kv_heads,
            head_dim,
            1.0,
            0.0,
        );
        let out_no_cap = b.download(&result_no_cap).as_f32().to_vec();

        // With softcap=1.0 (aggressive capping)
        let result_cap = b.grouped_attention_decode(
            &q,
            &k,
            &v,
            total_len,
            num_heads,
            num_kv_heads,
            head_dim,
            1.0,
            1.0,
        );
        let out_cap = b.download(&result_cap).as_f32().to_vec();

        // With softcap, the distribution should be more uniform
        // (scores are capped so the gap between pos 0 and pos 1 is smaller)
        let diff_no_cap = (out_no_cap[0] - out_no_cap[1]).abs();
        let diff_cap = (out_cap[0] - out_cap[1]).abs();
        assert!(
            diff_cap < diff_no_cap,
            "Softcap should make output more uniform: diff_no_cap={}, diff_cap={}",
            diff_no_cap,
            diff_cap
        );
    }

    #[test]
    fn test_grouped_attention_decode_single_token() {
        // With total_len=1, softmax gives probability 1.0 to the only position
        let b = backend();
        let q = dt(Tensor::new(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]));
        let k = dt(Tensor::new(vec![1, 4], vec![0.5, 0.5, 0.5, 0.5]));
        let v = dt(Tensor::new(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0]));

        let result = b.grouped_attention_decode(&q, &k, &v, 1, 2, 2, 2, 1.0 / (2.0f32).sqrt(), 0.0);

        let out = b.download(&result).as_f32().to_vec();
        // With a single position, output should equal V (softmax = 1.0)
        assert_close(
            &out,
            &[10.0, 20.0, 30.0, 40.0],
            1e-5,
            "single token attention",
        );
    }
}
