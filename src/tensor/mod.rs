//! N-dimensional tensor type with mixed dtype support including quantized formats.
//!
//! Provides the core [`Tensor`] type used throughout strata-inference. Supports F32, F16,
//! and quantized (Q8_0, Q4_0) storage formats with dequantization to F32.

use tracing::debug;

/// Data type of tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDtype {
    F32,
    F16,
    Q8_0,
    Q4_0,
}

impl TensorDtype {
    /// Number of values per quantization block (32 for Q8_0 and Q4_0).
    pub fn block_size(&self) -> usize {
        match self {
            TensorDtype::F32 => 1,
            TensorDtype::F16 => 1,
            TensorDtype::Q8_0 => 32,
            TensorDtype::Q4_0 => 32,
        }
    }

    /// Size in bytes of one quantization block.
    pub fn block_byte_size(&self) -> usize {
        match self {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            TensorDtype::Q8_0 => 34, // 2 bytes f16 scale + 32 bytes i8 values
            TensorDtype::Q4_0 => 18, // 2 bytes f16 scale + 16 bytes (32 nibbles)
        }
    }
}

/// Storage for tensor data, varying by dtype.
#[derive(Debug, Clone)]
pub enum TensorStorage {
    /// 32-bit floating point values.
    F32(Vec<f32>),
    /// 16-bit floating point values stored as raw u16 bits.
    F16(Vec<u16>),
    /// Raw quantized block data, interpreted according to the tensor's dtype.
    Quantized(Vec<u8>),
}

/// N-dimensional tensor with dtype and storage.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: TensorDtype,
    storage: TensorStorage,
}

/// Compute row-major strides from shape.
/// strides[i] = product of shape[i+1..]
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl Tensor {
    /// Create an F32 tensor from shape and data.
    ///
    /// # Panics
    /// Panics if `data.len()` does not match the product of `shape`.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n_elements: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n_elements,
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            n_elements
        );
        let strides = compute_strides(&shape);
        debug!(dtype = ?TensorDtype::F32, ?shape, "Created tensor");
        Self {
            shape,
            strides,
            dtype: TensorDtype::F32,
            storage: TensorStorage::F32(data),
        }
    }

    /// Create a zero-filled F32 tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        let n_elements: usize = shape.iter().product();
        let strides = compute_strides(shape);
        Self {
            shape: shape.to_vec(),
            strides,
            dtype: TensorDtype::F32,
            storage: TensorStorage::F32(vec![0.0f32; n_elements]),
        }
    }

    /// Create an F16 tensor from shape and raw u16 bit data.
    ///
    /// # Panics
    /// Panics if `data.len()` does not match the product of `shape`.
    pub fn from_f16(shape: Vec<usize>, data: Vec<u16>) -> Self {
        let n_elements: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n_elements,
            "F16 data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            n_elements
        );
        let strides = compute_strides(&shape);
        debug!(dtype = ?TensorDtype::F16, ?shape, "Created tensor");
        Self {
            shape,
            strides,
            dtype: TensorDtype::F16,
            storage: TensorStorage::F16(data),
        }
    }

    /// Create a quantized tensor from shape, dtype, and raw block data.
    ///
    /// # Panics
    /// Panics if `dtype` is not a quantized type (Q8_0 or Q4_0), or if the data length
    /// doesn't match the expected number of blocks.
    pub fn from_quantized(shape: Vec<usize>, dtype: TensorDtype, data: Vec<u8>) -> Self {
        assert!(
            dtype == TensorDtype::Q8_0 || dtype == TensorDtype::Q4_0,
            "from_quantized requires Q8_0 or Q4_0 dtype, got {:?}",
            dtype
        );
        let n_elements: usize = shape.iter().product();
        let block_size = dtype.block_size();
        let block_byte_size = dtype.block_byte_size();
        let n_blocks = (n_elements + block_size - 1) / block_size;
        let expected_bytes = n_blocks * block_byte_size;
        assert_eq!(
            data.len(),
            expected_bytes,
            "Quantized data length {} does not match expected {} bytes ({} blocks of {} bytes for {:?})",
            data.len(),
            expected_bytes,
            n_blocks,
            block_byte_size,
            dtype,
        );
        let strides = compute_strides(&shape);
        debug!(?dtype, ?shape, n_blocks, "Created quantized tensor");
        Self {
            shape,
            strides,
            dtype,
            storage: TensorStorage::Quantized(data),
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the strides of the tensor.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the data type of the tensor.
    pub fn dtype(&self) -> TensorDtype {
        self.dtype
    }

    /// Returns the storage of the tensor.
    pub fn storage(&self) -> &TensorStorage {
        &self.storage
    }

    /// Returns the total number of elements in the tensor.
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns a reference to the underlying F32 data.
    ///
    /// # Panics
    /// Panics if the tensor is not F32 dtype.
    pub fn as_f32(&self) -> &[f32] {
        match &self.storage {
            TensorStorage::F32(data) => data,
            _ => panic!("Tensor is {:?}, not F32", self.dtype),
        }
    }

    /// Returns a mutable reference to the underlying F32 data.
    ///
    /// # Panics
    /// Panics if the tensor is not F32 dtype.
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        match &mut self.storage {
            TensorStorage::F32(data) => data,
            _ => panic!("Tensor is {:?}, not F32", self.dtype),
        }
    }

    /// Convert the tensor to F32, dequantizing if necessary.
    ///
    /// - F32 tensors are cloned as-is.
    /// - F16 tensors are converted element-wise using the `half` crate.
    /// - Q8_0 tensors are dequantized: `y[i] = scale * qs[i]`
    /// - Q4_0 tensors are dequantized: `y[i] = (nibble - 8) * scale`
    pub fn to_f32(&self) -> Tensor {
        match self.dtype {
            TensorDtype::F32 => self.clone(),
            TensorDtype::F16 => {
                let data = match &self.storage {
                    TensorStorage::F16(bits) => bits
                        .iter()
                        .map(|&b| half::f16::from_bits(b).to_f32())
                        .collect(),
                    _ => unreachable!("F16 dtype must have F16 storage"),
                };
                Tensor {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    dtype: TensorDtype::F32,
                    storage: TensorStorage::F32(data),
                }
            }
            TensorDtype::Q8_0 => {
                let raw = match &self.storage {
                    TensorStorage::Quantized(data) => data,
                    _ => unreachable!("Q8_0 dtype must have Quantized storage"),
                };
                let n_elements = self.n_elements();
                let n_blocks = (n_elements + 31) / 32;
                let mut output = vec![0.0f32; n_elements];

                for block_idx in 0..n_blocks {
                    let block_start = block_idx * 34;
                    // First 2 bytes: f16 scale
                    let scale_bits =
                        u16::from_le_bytes([raw[block_start], raw[block_start + 1]]);
                    let scale = half::f16::from_bits(scale_bits).to_f32();

                    // Next 32 bytes: i8 quantized values
                    let values_in_block =
                        std::cmp::min(32, n_elements - block_idx * 32);
                    for i in 0..values_in_block {
                        let qs = raw[block_start + 2 + i] as i8;
                        output[block_idx * 32 + i] = scale * qs as f32;
                    }
                }

                Tensor {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    dtype: TensorDtype::F32,
                    storage: TensorStorage::F32(output),
                }
            }
            TensorDtype::Q4_0 => {
                let raw = match &self.storage {
                    TensorStorage::Quantized(data) => data,
                    _ => unreachable!("Q4_0 dtype must have Quantized storage"),
                };
                let n_elements = self.n_elements();
                let n_blocks = (n_elements + 31) / 32;
                let mut output = vec![0.0f32; n_elements];

                for block_idx in 0..n_blocks {
                    let block_start = block_idx * 18;
                    // First 2 bytes: f16 scale
                    let scale_bits =
                        u16::from_le_bytes([raw[block_start], raw[block_start + 1]]);
                    let scale = half::f16::from_bits(scale_bits).to_f32();

                    // Next 16 bytes: 32 4-bit values packed into 16 bytes.
                    // Layout matches llama.cpp's dequantize_row_q4_0:
                    //   low nibble  → first half  (indices 0..16)
                    //   high nibble → second half (indices 16..32)
                    let half = 16;
                    for j in 0..half {
                        let byte = raw[block_start + 2 + j];
                        let low = (byte & 0x0F) as i32 - 8;
                        let high = (byte >> 4) as i32 - 8;
                        let base = block_idx * 32;
                        if base + j < n_elements {
                            output[base + j] = low as f32 * scale;
                        }
                        if base + j + half < n_elements {
                            output[base + j + half] = high as f32 * scale;
                        }
                    }
                }

                Tensor {
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    dtype: TensorDtype::F32,
                    storage: TensorStorage::F32(output),
                }
            }
        }
    }

    /// Reshape the tensor to a new shape. The total number of elements must remain the same.
    /// Returns a new tensor with the same data but different shape.
    ///
    /// # Panics
    /// Panics if the new shape has a different number of elements.
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_n_elements: usize = new_shape.iter().product();
        assert_eq!(
            self.n_elements(),
            new_n_elements,
            "Cannot reshape tensor of {} elements to shape {:?} ({} elements)",
            self.n_elements(),
            new_shape,
            new_n_elements
        );
        let strides = compute_strides(new_shape);
        Tensor {
            shape: new_shape.to_vec(),
            strides,
            dtype: self.dtype,
            storage: self.storage.clone(),
        }
    }

    /// Returns the number of rows (first dimension) for a 2D tensor.
    ///
    /// # Panics
    /// Panics if the tensor is not 2D.
    pub fn rows(&self) -> usize {
        assert_eq!(
            self.shape.len(),
            2,
            "rows() requires a 2D tensor, got shape {:?}",
            self.shape
        );
        self.shape[0]
    }

    /// Returns the number of columns (second dimension) for a 2D tensor.
    ///
    /// # Panics
    /// Panics if the tensor is not 2D.
    pub fn cols(&self) -> usize {
        assert_eq!(
            self.shape.len(),
            2,
            "cols() requires a 2D tensor, got shape {:?}",
            self.shape
        );
        self.shape[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[3, 5]), vec![5, 1]);
        assert_eq!(compute_strides(&[10]), vec![1]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_new_f32_tensor() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), TensorDtype::F32);
        assert_eq!(t.n_elements(), 6);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.as_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_new_shape_mismatch() {
        Tensor::new(vec![2, 3], vec![1.0, 2.0]); // only 2 elements, need 6
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.n_elements(), 12);
        assert!(t.as_f32().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_zeros_empty() {
        let t = Tensor::zeros(&[0]);
        assert_eq!(t.n_elements(), 0);
        assert!(t.as_f32().is_empty());
    }

    #[test]
    fn test_from_f16() {
        // f16 for 1.0 = 0x3C00, 2.0 = 0x4000
        let one = half::f16::from_f32(1.0).to_bits();
        let two = half::f16::from_f32(2.0).to_bits();
        let t = Tensor::from_f16(vec![2], vec![one, two]);
        assert_eq!(t.dtype(), TensorDtype::F16);
        assert_eq!(t.n_elements(), 2);
    }

    #[test]
    fn test_f16_to_f32() {
        let vals: Vec<f32> = vec![1.0, -0.5, 3.14, 0.0];
        let bits: Vec<u16> = vals.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let t = Tensor::from_f16(vec![4], bits);
        let converted = t.to_f32();
        assert_eq!(converted.dtype(), TensorDtype::F32);
        let data = converted.as_f32();
        for (i, &expected) in vals.iter().enumerate() {
            let diff = (data[i] - expected).abs();
            assert!(diff < 0.01, "F16 round-trip failed at index {}: expected {}, got {}", i, expected, data[i]);
        }
    }

    #[test]
    fn test_q8_0_dequantize() {
        // Build a Q8_0 block: scale=0.5 (f16), then 32 i8 values [0, 1, 2, ..., 31]
        let scale = half::f16::from_f32(0.5);
        let scale_bytes = scale.to_bits().to_le_bytes();
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale_bytes);
        for i in 0..32u8 {
            block.push(i); // treated as i8, all positive here
        }
        assert_eq!(block.len(), 34);

        let t = Tensor::from_quantized(vec![32], TensorDtype::Q8_0, block);
        let f32_t = t.to_f32();
        let data = f32_t.as_f32();
        assert_eq!(data.len(), 32);
        for i in 0..32 {
            let expected = 0.5 * i as f32;
            assert!(
                (data[i] - expected).abs() < 1e-6,
                "Q8_0 dequant failed at {}: expected {}, got {}",
                i,
                expected,
                data[i]
            );
        }
    }

    #[test]
    fn test_q8_0_negative_values() {
        // Test with negative i8 values
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale_bytes);
        // -1 as i8 = 0xFF as u8
        for _ in 0..32 {
            block.push(0xFF); // -1 as i8
        }

        let t = Tensor::from_quantized(vec![32], TensorDtype::Q8_0, block);
        let f32_t = t.to_f32();
        let data = f32_t.as_f32();
        for i in 0..32 {
            assert!(
                (data[i] - (-1.0)).abs() < 1e-6,
                "Q8_0 negative dequant failed at {}: expected -1.0, got {}",
                i,
                data[i]
            );
        }
    }

    #[test]
    fn test_q4_0_dequantize() {
        // Build a Q4_0 block: scale=1.0 (f16), then 16 bytes of packed nibbles.
        // Layout matches llama.cpp: low nibbles → first half (0..16),
        // high nibbles → second half (16..32).
        let scale = half::f16::from_f32(1.0);
        let scale_bytes = scale.to_bits().to_le_bytes();
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale_bytes);

        // byte[j] = j | (j << 4)  →  low nibble = j, high nibble = j
        for j in 0..16u8 {
            block.push(j | (j << 4));
        }
        assert_eq!(block.len(), 18);

        let t = Tensor::from_quantized(vec![32], TensorDtype::Q4_0, block);
        let f32_t = t.to_f32();
        let data = f32_t.as_f32();
        assert_eq!(data.len(), 32);

        // First half (indices 0..16): low nibbles = 0,1,2,...,15
        // Second half (indices 16..32): high nibbles = 0,1,2,...,15
        for j in 0..16 {
            let expected = (j as f32 - 8.0) * 1.0;
            assert!(
                (data[j] - expected).abs() < 1e-6,
                "Q4_0 low nibble at {}: expected {}, got {}",
                j, expected, data[j]
            );
            assert!(
                (data[j + 16] - expected).abs() < 1e-6,
                "Q4_0 high nibble at {}: expected {}, got {}",
                j + 16, expected, data[j + 16]
            );
        }
    }

    #[test]
    fn test_q8_0_multiple_blocks() {
        // Two blocks of 32 values each = 64 elements
        let scale1 = half::f16::from_f32(0.25);
        let scale2 = half::f16::from_f32(0.5);
        let mut data = Vec::new();

        // Block 1
        data.extend_from_slice(&scale1.to_bits().to_le_bytes());
        for i in 0..32u8 {
            data.push(i);
        }
        // Block 2
        data.extend_from_slice(&scale2.to_bits().to_le_bytes());
        for i in 0..32u8 {
            data.push(i);
        }

        let t = Tensor::from_quantized(vec![64], TensorDtype::Q8_0, data);
        let f32_t = t.to_f32();
        let out = f32_t.as_f32();
        assert_eq!(out.len(), 64);

        // First block: scale=0.25
        for i in 0..32 {
            let expected = 0.25 * i as f32;
            assert!((out[i] - expected).abs() < 1e-4, "Block 1 index {}", i);
        }
        // Second block: scale=0.5
        for i in 0..32 {
            let expected = 0.5 * i as f32;
            assert!((out[32 + i] - expected).abs() < 1e-4, "Block 2 index {}", i);
        }
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = t.reshape(&[3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.n_elements(), 6);
        assert_eq!(reshaped.as_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(reshaped.strides(), &[2, 1]);
    }

    #[test]
    fn test_reshape_to_1d() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let flat = t.reshape(&[6]);
        assert_eq!(flat.shape(), &[6]);
    }

    #[test]
    #[should_panic(expected = "Cannot reshape")]
    fn test_reshape_wrong_size() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        t.reshape(&[2, 2]); // 4 != 6
    }

    #[test]
    fn test_rows_cols() {
        let t = Tensor::new(vec![3, 5], vec![0.0; 15]);
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 5);
    }

    #[test]
    #[should_panic(expected = "rows() requires a 2D tensor")]
    fn test_rows_not_2d() {
        let t = Tensor::new(vec![3, 4, 5], vec![0.0; 60]);
        t.rows();
    }

    #[test]
    #[should_panic(expected = "cols() requires a 2D tensor")]
    fn test_cols_not_2d() {
        let t = Tensor::new(vec![3], vec![0.0; 3]);
        t.cols();
    }

    #[test]
    fn test_as_f32_mut() {
        let mut t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        t.as_f32_mut()[1] = 99.0;
        assert_eq!(t.as_f32(), &[1.0, 99.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "not F32")]
    fn test_as_f32_wrong_type() {
        let bits = vec![half::f16::from_f32(1.0).to_bits()];
        let t = Tensor::from_f16(vec![1], bits);
        t.as_f32(); // should panic
    }

    #[test]
    fn test_f32_to_f32_is_clone() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let t2 = t.to_f32();
        assert_eq!(t2.as_f32(), t.as_f32());
    }

    #[test]
    fn test_single_element_tensor() {
        let t = Tensor::new(vec![1], vec![42.0]);
        assert_eq!(t.n_elements(), 1);
        assert_eq!(t.as_f32(), &[42.0]);
    }

    #[test]
    fn test_high_dimensional_tensor() {
        let t = Tensor::zeros(&[2, 3, 4, 5]);
        assert_eq!(t.n_elements(), 120);
        assert_eq!(t.strides(), &[60, 20, 5, 1]);
    }

    #[test]
    fn test_q8_0_2d_shape() {
        // Create a 2x32 Q8_0 tensor (2 rows, each row is one block of 32)
        let scale = half::f16::from_f32(1.0);
        let mut data = Vec::new();
        for _ in 0..2 {
            data.extend_from_slice(&scale.to_bits().to_le_bytes());
            for i in 0..32u8 {
                data.push(i);
            }
        }
        let t = Tensor::from_quantized(vec![2, 32], TensorDtype::Q8_0, data);
        assert_eq!(t.shape(), &[2, 32]);
        assert_eq!(t.n_elements(), 64);
        let f32_t = t.to_f32();
        assert_eq!(f32_t.shape(), &[2, 32]);
    }

    // ====================================================================
    // NEW TESTS: Tensor reshaping, edge cases, dtype properties
    // ====================================================================

    // -- Reshape edge cases --

    #[test]
    fn test_reshape_same_shape() {
        // Reshaping to the same shape should work and return identical data.
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reshaped = t.reshape(&[2, 3]);
        assert_eq!(reshaped.shape(), &[2, 3]);
        assert_eq!(reshaped.as_f32(), t.as_f32());
    }

    #[test]
    fn test_reshape_1d_to_2d_to_3d() {
        // Test 1D -> 2D -> 3D and back.
        let t = Tensor::new(vec![24], (1..=24).map(|i| i as f32).collect());
        let t2d = t.reshape(&[4, 6]);
        assert_eq!(t2d.shape(), &[4, 6]);
        assert_eq!(t2d.strides(), &[6, 1]);
        assert_eq!(t2d.as_f32()[0], 1.0);
        assert_eq!(t2d.as_f32()[23], 24.0);

        let t3d = t2d.reshape(&[2, 3, 4]);
        assert_eq!(t3d.shape(), &[2, 3, 4]);
        assert_eq!(t3d.strides(), &[12, 4, 1]);
        assert_eq!(t3d.as_f32()[0], 1.0);
        assert_eq!(t3d.as_f32()[23], 24.0);

        let t1d = t3d.reshape(&[24]);
        assert_eq!(t1d.shape(), &[24]);
        assert_eq!(t1d.as_f32(), t.as_f32());
    }

    #[test]
    fn test_reshape_preserves_dtype() {
        let t = Tensor::new(vec![6], vec![1.0; 6]);
        let reshaped = t.reshape(&[2, 3]);
        assert_eq!(reshaped.dtype(), TensorDtype::F32);
    }

    // -- TensorDtype properties --

    #[test]
    fn test_tensor_dtype_block_sizes() {
        assert_eq!(TensorDtype::F32.block_size(), 1);
        assert_eq!(TensorDtype::F16.block_size(), 1);
        assert_eq!(TensorDtype::Q8_0.block_size(), 32);
        assert_eq!(TensorDtype::Q4_0.block_size(), 32);
    }

    #[test]
    fn test_tensor_dtype_block_byte_sizes() {
        assert_eq!(TensorDtype::F32.block_byte_size(), 4);
        assert_eq!(TensorDtype::F16.block_byte_size(), 2);
        assert_eq!(TensorDtype::Q8_0.block_byte_size(), 34);
        assert_eq!(TensorDtype::Q4_0.block_byte_size(), 18);
    }

    // -- from_quantized panics --

    #[test]
    #[should_panic(expected = "from_quantized requires Q8_0 or Q4_0")]
    fn test_from_quantized_wrong_dtype() {
        Tensor::from_quantized(vec![4], TensorDtype::F32, vec![0u8; 16]);
    }

    #[test]
    #[should_panic(expected = "Quantized data length")]
    fn test_from_quantized_wrong_data_size() {
        // 32 elements need 34 bytes for Q8_0, but we provide only 10.
        Tensor::from_quantized(vec![32], TensorDtype::Q8_0, vec![0u8; 10]);
    }

    // -- F16 tensor conversion --

    #[test]
    fn test_f16_tensor_zero() {
        let zero = half::f16::from_f32(0.0).to_bits();
        let t = Tensor::from_f16(vec![1], vec![zero]);
        let f32_t = t.to_f32();
        assert_eq!(f32_t.as_f32(), &[0.0]);
    }

    #[test]
    fn test_f16_tensor_negative() {
        let neg = half::f16::from_f32(-3.5).to_bits();
        let t = Tensor::from_f16(vec![1], vec![neg]);
        let f32_t = t.to_f32();
        assert!((f32_t.as_f32()[0] - (-3.5)).abs() < 0.01);
    }

    // -- compute_strides --

    #[test]
    fn test_compute_strides_1d() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_compute_strides_single_element() {
        assert_eq!(compute_strides(&[1, 1, 1]), vec![1, 1, 1]);
    }

    // -- Tensor storage access --

    #[test]
    fn test_storage_variant() {
        let t = Tensor::new(vec![2], vec![1.0, 2.0]);
        assert!(matches!(t.storage(), TensorStorage::F32(_)));

        let bits = vec![half::f16::from_f32(1.0).to_bits()];
        let t = Tensor::from_f16(vec![1], bits);
        assert!(matches!(t.storage(), TensorStorage::F16(_)));

        let scale = half::f16::from_f32(1.0);
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        data.extend_from_slice(&[0u8; 32]);
        let t = Tensor::from_quantized(vec![32], TensorDtype::Q8_0, data);
        assert!(matches!(t.storage(), TensorStorage::Quantized(_)));
    }

    // -- Q4_0 tensor dequantization with known pattern --

    #[test]
    fn test_q4_0_neutral_dequant() {
        // Neutral nibbles (0x88 = both nibbles = 8) should produce zeros.
        let scale = half::f16::from_f32(1.0);
        let mut data = Vec::new();
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        data.extend_from_slice(&[0x88u8; 16]);

        let t = Tensor::from_quantized(vec![32], TensorDtype::Q4_0, data);
        let f32_t = t.to_f32();
        for &v in f32_t.as_f32() {
            assert!(v.abs() < 1e-6, "expected 0.0, got {}", v);
        }
    }

    // -- Clone preserves data --

    #[test]
    fn test_tensor_clone() {
        let t = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let cloned = t.clone();
        assert_eq!(cloned.shape(), t.shape());
        assert_eq!(cloned.as_f32(), t.as_f32());
        assert_eq!(cloned.dtype(), t.dtype());
    }

    // -- as_f32_mut on non-F32 panics --

    #[test]
    #[should_panic(expected = "not F32")]
    fn test_as_f32_mut_wrong_type() {
        let bits = vec![half::f16::from_f32(1.0).to_bits()];
        let mut t = Tensor::from_f16(vec![1], bits);
        t.as_f32_mut(); // should panic
    }

    // -- Large tensor --

    #[test]
    fn test_large_tensor_zeros() {
        // 1024 elements
        let t = Tensor::zeros(&[32, 32]);
        assert_eq!(t.n_elements(), 1024);
        assert!(t.as_f32().iter().all(|&v| v == 0.0));
    }
}
