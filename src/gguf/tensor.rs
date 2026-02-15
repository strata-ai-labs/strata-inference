// M1: Tensor data loading and memory mapping from GGUF files

use crate::error::InferenceError;
use super::GgufFile;
use super::quant::{
    self, GgufTensorType, bytes_as_q4_0_blocks, bytes_as_q8_0_blocks,
    dequantize_q4_0, dequantize_q8_0, f16_to_f32,
};

// ---------------------------------------------------------------------------
// GgufTensor — a named, typed view into memory-mapped tensor data
// ---------------------------------------------------------------------------

/// A tensor loaded from a GGUF file.
///
/// The raw data is a zero-copy slice from the memory-mapped file. Use the
/// `to_f32()` method to dequantize into a `Vec<f32>` for computation.
#[derive(Debug)]
pub struct GgufTensor<'a> {
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    pub name: String,
    /// Shape as stored in the GGUF file (GGUF uses row-major innermost-first,
    /// i.e. dims[0] is the "fastest" dimension / number of columns).
    pub shape: Vec<u64>,
    /// Data type / quantization format.
    pub dtype: GgufTensorType,
    /// Raw bytes of the tensor data (zero-copy from mmap).
    pub data: &'a [u8],
}

impl<'a> GgufTensor<'a> {
    /// Total number of logical elements in the tensor.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// Size in bytes of the raw data slice.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Expected byte size computed from shape and dtype.
    pub fn expected_byte_size(&self) -> u64 {
        quant::tensor_byte_size(self.dtype, self.n_elements())
    }

    /// Dequantize the tensor data to f32.
    ///
    /// Supported types: F32, F16, Q8_0, Q4_0.
    /// Returns an error for unsupported quantization types.
    pub fn to_f32(&self) -> Result<Vec<f32>, InferenceError> {
        match self.dtype {
            GgufTensorType::F32 => {
                // Direct reinterpretation — every 4 bytes is one LE f32.
                let n = self.n_elements() as usize;
                let mut result = Vec::with_capacity(n);
                for chunk in self.data.chunks_exact(4) {
                    let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    result.push(f32::from_le_bytes(bytes));
                }
                Ok(result)
            }
            GgufTensorType::F16 => {
                // Every 2 bytes is one f16 (LE).
                let n = self.n_elements() as usize;
                let mut result = Vec::with_capacity(n);
                for chunk in self.data.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    result.push(f16_to_f32(bits));
                }
                Ok(result)
            }
            GgufTensorType::Q8_0 => {
                let blocks = bytes_as_q8_0_blocks(self.data)?;
                Ok(dequantize_q8_0(blocks))
            }
            GgufTensorType::Q4_0 => {
                let blocks = bytes_as_q4_0_blocks(self.data)?;
                Ok(dequantize_q4_0(blocks))
            }
            other => Err(InferenceError::GgufParse(format!(
                "dequantization not yet implemented for type {}",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Loading tensors from GgufFile
// ---------------------------------------------------------------------------

/// Load a tensor by index from a parsed GGUF file.
///
/// The returned `GgufTensor` borrows directly from the file's memory map
/// (zero-copy). The tensor data is validated for size consistency.
pub fn load_tensor<'a>(
    file: &'a GgufFile,
    index: usize,
) -> Result<GgufTensor<'a>, InferenceError> {
    let infos = file.tensor_infos();
    let info = infos.get(index).ok_or_else(|| {
        InferenceError::TensorNotFound(format!("tensor index {} out of range", index))
    })?;

    let data = file.tensor_data(index)?;

    let expected_size = info.byte_size() as usize;
    if data.len() != expected_size {
        return Err(InferenceError::GgufParse(format!(
            "tensor '{}': expected {} bytes, got {} bytes",
            info.name,
            expected_size,
            data.len()
        )));
    }

    Ok(GgufTensor {
        name: info.name.clone(),
        shape: info.dims.clone(),
        dtype: info.dtype,
        data,
    })
}

/// Load a tensor by name from a parsed GGUF file.
///
/// Returns an error if the tensor is not found.
pub fn load_tensor_by_name<'a>(
    file: &'a GgufFile,
    name: &str,
) -> Result<GgufTensor<'a>, InferenceError> {
    let (index, _) = file
        .find_tensor(name)
        .ok_or_else(|| InferenceError::TensorNotFound(name.to_string()))?;
    load_tensor(file, index)
}

/// Load all tensors from a parsed GGUF file.
///
/// Returns a Vec of `GgufTensor` in the order they appear in the file.
pub fn load_all_tensors(file: &GgufFile) -> Result<Vec<GgufTensor<'_>>, InferenceError> {
    let mut tensors = Vec::with_capacity(file.n_tensors());
    for i in 0..file.n_tensors() {
        tensors.push(load_tensor(file, i)?);
    }
    Ok(tensors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::quant::f32_to_f16;

    /// Helper: build a minimal GGUF file with one tensor and return the path.
    fn build_gguf_with_tensor(
        tensor_name: &str,
        dims: &[u64],
        dtype_id: u32,
        tensor_data: &[u8],
    ) -> (std::path::PathBuf, Vec<u8>) {
        let magic: u32 = 0x4655_4747;
        let alignment: u32 = 32;

        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&magic.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv

        // KV: general.architecture = "test"
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING type
        let val = "test";
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());

        // Tensor info
        buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
        buf.extend_from_slice(tensor_name.as_bytes());
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&dtype_id.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // Pad to alignment
        let pos = buf.len();
        let aligned = ((pos + alignment as usize - 1) / alignment as usize) * alignment as usize;
        buf.resize(aligned, 0);

        // Tensor data
        buf.extend_from_slice(tensor_data);

        // Pad to alignment
        let pos = buf.len();
        let aligned = ((pos + alignment as usize - 1) / alignment as usize) * alignment as usize;
        buf.resize(aligned, 0);

        let dir = std::env::temp_dir();
        let path = dir.join(format!("test_tensor_{}.gguf", tensor_name));
        std::fs::write(&path, &buf).unwrap();

        (path, buf)
    }

    #[test]
    fn test_load_f32_tensor() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (path, _) = build_gguf_with_tensor("f32_tensor", &[4], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.name, "f32_tensor");
        assert_eq!(tensor.shape, vec![4]);
        assert_eq!(tensor.dtype, GgufTensorType::F32);
        assert_eq!(tensor.n_elements(), 4);
        assert_eq!(tensor.byte_size(), 16);

        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data.len(), 4);
        assert_eq!(f32_data, vec![1.0, 2.0, 3.0, 4.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_f16_tensor() {
        let values = [1.0f32, -0.5, 0.0, 2.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        let (path, _) = build_gguf_with_tensor("f16_tensor", &[4], 1, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.dtype, GgufTensorType::F16);

        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data.len(), 4);
        assert!((f32_data[0] - 1.0).abs() < 1e-3);
        assert!((f32_data[1] - (-0.5)).abs() < 1e-3);
        assert!((f32_data[2] - 0.0).abs() < 1e-3);
        assert!((f32_data[3] - 2.0).abs() < 1e-2);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_q8_0_tensor() {
        // Build one Q8_0 block: d=1.0 as f16, qs=[1..=32] (well, [1..32])
        let d_bits = f32_to_f16(1.0);
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&d_bits.to_le_bytes());
        for i in 1..=32i8 {
            block_data.push(i as u8);
        }
        assert_eq!(block_data.len(), 34);

        let (path, _) = build_gguf_with_tensor("q8_tensor", &[32], 8, &block_data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.dtype, GgufTensorType::Q8_0);
        assert_eq!(tensor.n_elements(), 32);

        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data.len(), 32);
        // d=1.0 * qs[0]=1 = 1.0
        assert!((f32_data[0] - 1.0).abs() < 1e-3);
        // d=1.0 * qs[31]=32 = 32.0
        assert!((f32_data[31] - 32.0).abs() < 0.1);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_q4_0_tensor() {
        // Build one Q4_0 block: d=1.0 as f16, qs=[0x88; 16] (all zeros after dequant)
        let d_bits = f32_to_f16(1.0);
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&d_bits.to_le_bytes());
        block_data.extend_from_slice(&[0x88u8; 16]);
        assert_eq!(block_data.len(), 18);

        let (path, _) = build_gguf_with_tensor("q4_tensor", &[32], 2, &block_data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.dtype, GgufTensorType::Q4_0);

        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data.len(), 32);
        // All values should be zero (nibble 8 - 8 = 0)
        for &v in &f32_data {
            assert!(v.abs() < 1e-6, "expected ~0.0, got {}", v);
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_by_name() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (path, _) = build_gguf_with_tensor("named_tensor", &[4], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor_by_name(&file, "named_tensor").unwrap();
        assert_eq!(tensor.name, "named_tensor");

        // Non-existent
        assert!(load_tensor_by_name(&file, "nonexistent").is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_all_tensors() {
        // Build a file with one tensor
        let data: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (path, _) = build_gguf_with_tensor("all_tensor", &[2], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensors = load_all_tensors(&file).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "all_tensor");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_tensor_shape_2d() {
        // 2D F32 tensor: [4, 3] = 12 elements = 48 bytes
        let data: Vec<u8> = (1..=12)
            .map(|i| i as f32)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (path, _) = build_gguf_with_tensor("matrix", &[4, 3], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.shape, vec![4, 3]);
        assert_eq!(tensor.n_elements(), 12);

        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data.len(), 12);
        assert_eq!(f32_data[0], 1.0);
        assert_eq!(f32_data[11], 12.0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_expected_byte_size() {
        let data = vec![0u8; 16];
        let (path, _) = build_gguf_with_tensor("size_test", &[4], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.expected_byte_size(), 16);
        assert_eq!(tensor.byte_size(), 16);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unsupported_dequant_type() {
        // Q4_1 (type=3) is not yet supported for dequantization
        // Q4_1 block size: 20 bytes for 32 elements
        let data = vec![0u8; 20]; // 1 block
        let (path, _) = build_gguf_with_tensor("q4_1_tensor", &[32], 3, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        let result = tensor.to_f32();
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    // ====================================================================
    // NEW TESTS: Tensor loading edge cases and extended coverage
    // ====================================================================

    #[test]
    fn test_load_tensor_index_out_of_range() {
        // Trying to load a tensor at an invalid index should fail.
        let data = vec![0u8; 16]; // 4 F32 values
        let (path, _) = build_gguf_with_tensor("test_oob", &[4], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let result = load_tensor(&file, 99);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_tensor_n_elements_and_sizes() {
        // Verify n_elements, byte_size, and expected_byte_size for various shapes.
        let data: Vec<u8> = (1..=24).map(|i| i as f32).flat_map(|v| v.to_le_bytes()).collect();
        let (path, _) = build_gguf_with_tensor("matrix24", &[6, 4], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.n_elements(), 24);
        assert_eq!(tensor.byte_size(), 96); // 24 * 4 bytes
        assert_eq!(tensor.expected_byte_size(), 96);
        assert_eq!(tensor.shape, vec![6, 4]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_f32_tensor_values_roundtrip() {
        // Load specific f32 values and verify all of them round-trip correctly.
        let values = [-1.0f32, 0.0, 1.0, 3.14, -2.71, 1e-6, 1e6, f32::MAX];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (path, _) = build_gguf_with_tensor("roundtrip", &[8], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        let f32_data = tensor.to_f32().unwrap();
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(
                f32_data[i], expected,
                "mismatch at index {}: got {}, expected {}",
                i, f32_data[i], expected
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_q8_0_tensor_precision() {
        // Build a Q8_0 block with scale=0.125 and values [-4, -3, ..., 27]
        // and verify dequantization matches expected values within f16 tolerance.
        let d_bits = f32_to_f16(0.125);
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&d_bits.to_le_bytes());
        for i in 0..32i8 {
            let val = i - 4; // range: -4 to 27
            block_data.push(val as u8);
        }

        let (path, _) = build_gguf_with_tensor("q8_precision", &[32], 8, &block_data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        let f32_data = tensor.to_f32().unwrap();

        let scale = half::f16::from_bits(d_bits).to_f32();
        for i in 0..32 {
            let qs_val = (i as i8) - 4;
            let expected = scale * qs_val as f32;
            assert!(
                (f32_data[i] - expected).abs() < 1e-6,
                "Q8_0 precision mismatch at {}: got {}, expected {}",
                i, f32_data[i], expected
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_f16_tensor_all_special_values() {
        // Test f16 tensor with zero, negative, and small values.
        let values = [0.0f32, -1.0, 0.5, -0.5];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect();
        let (path, _) = build_gguf_with_tensor("f16_special", &[4], 1, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        let f32_data = tensor.to_f32().unwrap();

        for (i, &expected) in values.iter().enumerate() {
            assert!(
                (f32_data[i] - expected).abs() < 1e-3,
                "F16 special value mismatch at {}: got {}, expected {}",
                i, f32_data[i], expected
            );
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_3d_shape() {
        // Verify that 3D tensors can be loaded.
        // 2x3x2 = 12 elements = 48 bytes
        let data: Vec<u8> = (1..=12).map(|i| i as f32).flat_map(|v| v.to_le_bytes()).collect();
        let (path, _) = build_gguf_with_tensor("tensor3d", &[2, 3, 2], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.shape, vec![2, 3, 2]);
        assert_eq!(tensor.n_elements(), 12);
        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data.len(), 12);
        assert_eq!(f32_data[0], 1.0);
        assert_eq!(f32_data[11], 12.0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_1d_single_element() {
        // A 1D tensor with a single element.
        let data: Vec<u8> = 42.0f32.to_le_bytes().to_vec();
        let (path, _) = build_gguf_with_tensor("scalar_like", &[1], 0, &data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        assert_eq!(tensor.n_elements(), 1);
        let f32_data = tensor.to_f32().unwrap();
        assert_eq!(f32_data, vec![42.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_tensor_q4_0_known_pattern() {
        // Build a Q4_0 block where low=10, high=6 for byte index 0
        // and all others neutral (0x88).
        // d=2.0, low nibble=10: (10-8)*2.0=4.0, high nibble=6: (6-8)*2.0=-4.0
        let d_bits = f32_to_f16(2.0);
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&d_bits.to_le_bytes());
        let mut qs = [0x88u8; 16];
        qs[0] = 0x6A; // low=0xA=10, high=0x6=6
        block_data.extend_from_slice(&qs);

        let (path, _) = build_gguf_with_tensor("q4_known", &[32], 2, &block_data);

        let file = GgufFile::open(&path).unwrap();
        let tensor = load_tensor(&file, 0).unwrap();
        let f32_data = tensor.to_f32().unwrap();

        // index 0 (low nibble of byte 0): (10 - 8) * 2.0 = 4.0
        assert!((f32_data[0] - 4.0).abs() < 1e-3, "got {}", f32_data[0]);
        // index 16 (high nibble of byte 0): (6 - 8) * 2.0 = -4.0
        assert!((f32_data[16] - (-4.0)).abs() < 1e-3, "got {}", f32_data[16]);
        // index 1 (low nibble of byte 1 = 0x88): (8 - 8) * 2.0 = 0.0
        assert!((f32_data[1]).abs() < 1e-3, "got {}", f32_data[1]);

        std::fs::remove_file(&path).ok();
    }
}
