// M1: Quantization block formats and dequantization (Q8_0, Q4_0, F16, F32)

use crate::error::InferenceError;

// ---------------------------------------------------------------------------
// GGML tensor type IDs (matches ggml_type enum in ggml.h)
// ---------------------------------------------------------------------------

/// Tensor data types supported by GGUF files.
///
/// The discriminant values match the GGML type IDs from the GGUF spec.
/// Only the types we actually need are fully implemented; others are listed
/// so we can at least *parse* the type field without erroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4 (removed)
    // Q4_3 = 5 (removed)
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GgufTensorType {
    /// Convert a raw u32 from the GGUF file into a `GgufTensorType`.
    pub fn from_u32(v: u32) -> Result<Self, InferenceError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::IQ2XXS),
            17 => Ok(Self::IQ2XS),
            18 => Ok(Self::IQ3XXS),
            19 => Ok(Self::IQ1S),
            20 => Ok(Self::IQ4NL),
            21 => Ok(Self::IQ3S),
            22 => Ok(Self::IQ2S),
            23 => Ok(Self::IQ4XS),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::IQ1M),
            30 => Ok(Self::BF16),
            _ => Err(InferenceError::UnknownTensorType(v)),
        }
    }

    /// Number of elements per quantization block.
    ///
    /// For non-quantized types (F32, F16, etc.) the block size is 1.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            // K-quant super-blocks use 256 elements
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ3XXS | Self::IQ1S
            | Self::IQ4NL | Self::IQ3S | Self::IQ2S | Self::IQ4XS | Self::IQ1M => 256,
        }
    }

    /// Size in bytes of one quantization block.
    ///
    /// For non-quantized types this is the size of a single element.
    pub fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(), // 18
            Self::Q4_1 => 20,   // 2*f16 + 16 bytes
            Self::Q5_0 => 22,   // f16 + 4 + 16 bytes
            Self::Q5_1 => 24,   // 2*f16 + 4 + 16 bytes
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(), // 34
            Self::Q8_1 => 36,   // 2*f16 + 32 bytes
            // K-quant sizes (QK_K = 256)
            Self::Q2K => 2 * 2 + 256 / 16 + 256 / 4,       // 84
            Self::Q3K => 2 + 256 / 4 + 256 / 8 + 12,       // 110
            Self::Q4K => 2 * 2 + 12 + 256 / 2,              // 144
            Self::Q5K => 2 * 2 + 12 + 256 / 8 + 256 / 2,    // 176
            Self::Q6K => 2 + 256 / 16 + 3 * 256 / 4,        // 210
            Self::Q8K => 4 + 256 + 256 / 16 * 2,            // 292
            // IQ types — approximate; not fully implemented for inference yet
            Self::IQ2XXS => 2 + 256 / 8 * 2,
            Self::IQ2XS => 2 + 256 / 8 * 2 + 256 / 32,
            Self::IQ3XXS => 2 + 3 * (256 / 8),
            Self::IQ1S => 2 + 256 / 8 + 256 / 16,
            Self::IQ4NL => 2 + 256 / 2,
            Self::IQ3S => 2 + 256 / 4 + 256 / 32 + 256 / 8 + 256 / 64,
            Self::IQ2S => 2 + 256 / 4 + 256 / 16,
            Self::IQ4XS => 2 + 256 / 64 + 256 / 2,
            Self::IQ1M => 2 + 256 / 8 + 256 / 16,
        }
    }

    /// Human-readable name for the tensor type.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ1S => "IQ1_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ2S => "IQ2_S",
            Self::IQ4XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1M => "IQ1_M",
            Self::BF16 => "BF16",
        }
    }
}

impl std::fmt::Display for GgufTensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Q8_0 quantization block: 34 bytes per block of 32 values
// ---------------------------------------------------------------------------

/// Q8_0 block: 8-bit quantization with a single f16 scale factor.
///
/// Layout: `d: f16 (2 bytes) | qs: [i8; 32] (32 bytes)` = 34 bytes total.
/// Dequantization: `y[i] = f16_to_f32(d) * qs[i]`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    /// Scale factor stored as IEEE 754 half-precision (f16) bits.
    pub d: u16,
    /// 32 quantized values, each in [-127, 127].
    pub qs: [i8; 32],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

/// Number of elements per Q8_0 block.
pub const QK8_0: usize = 32;

// ---------------------------------------------------------------------------
// Q4_0 quantization block: 18 bytes per block of 32 values
// ---------------------------------------------------------------------------

/// Q4_0 block: 4-bit quantization with a single f16 scale factor.
///
/// Layout: `d: f16 (2 bytes) | qs: [u8; 16] (16 bytes)` = 18 bytes total.
/// Each byte in `qs` stores two 4-bit values (low nibble first, high nibble second).
/// Dequantization: `y[j]       = (low_nibble  - 8) * d`
///                  `y[j + 16]  = (high_nibble - 8) * d`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    /// Scale factor stored as IEEE 754 half-precision (f16) bits.
    pub d: u16,
    /// 16 bytes of packed 4-bit quantized values (32 values total).
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

/// Number of elements per Q4_0 block.
pub const QK4_0: usize = 32;

// ---------------------------------------------------------------------------
// IEEE 754 half-precision (f16) conversion
// ---------------------------------------------------------------------------

/// Convert a 16-bit IEEE 754 half-precision float to a 32-bit float.
///
/// Bit layout of f16:
///   - 1 bit sign
///   - 5 bits exponent (bias 15)
///   - 10 bits mantissa
///
/// Special cases:
///   - Exponent 0: subnormal (or zero)
///   - Exponent 31: infinity / NaN
pub fn f16_to_f32(bits: u16) -> f32 {
    // Use the `half` crate for correctness (it handles subnormals, inf, nan).
    half::f16::from_bits(bits).to_f32()
}

/// Convert a 32-bit float to 16-bit IEEE 754 half-precision bits.
pub fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

// ---------------------------------------------------------------------------
// Dequantization functions
// ---------------------------------------------------------------------------

/// Dequantize a slice of Q8_0 blocks into f32 values.
///
/// Each block of 32 quantized i8 values is scaled by its f16 delta `d`.
/// Output length = `blocks.len() * 32`.
pub fn dequantize_q8_0(blocks: &[BlockQ8_0]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK8_0);
    for block in blocks {
        let d = f16_to_f32(block.d);
        for &q in &block.qs {
            output.push(d * q as f32);
        }
    }
    output
}

/// Dequantize a slice of Q4_0 blocks into f32 values.
///
/// Each byte contains two 4-bit nibbles. The low nibble maps to the first
/// half of the block (indices 0..15) and the high nibble to the second half
/// (indices 16..31). Each nibble is an unsigned value in [0, 15] that is
/// shifted to signed by subtracting 8, then scaled by `d`.
///
/// Output length = `blocks.len() * 32`.
pub fn dequantize_q4_0(blocks: &[BlockQ4_0]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK4_0);
    for block in blocks {
        let d = f16_to_f32(block.d);
        // First pass: low nibbles -> first 16 values
        // Second pass: high nibbles -> next 16 values
        // Following the exact layout from llama.cpp's dequantize_row_q4_0:
        //   y[i*qk + j + 0]    = (qs[j] & 0x0F) - 8) * d
        //   y[i*qk + j + qk/2] = (qs[j] >> 4)   - 8) * d
        let mut tmp = [0.0f32; QK4_0];
        for j in 0..QK4_0 / 2 {
            let low = (block.qs[j] & 0x0F) as i32 - 8;
            let high = (block.qs[j] >> 4) as i32 - 8;
            tmp[j] = low as f32 * d;
            tmp[j + QK4_0 / 2] = high as f32 * d;
        }
        output.extend_from_slice(&tmp);
    }
    output
}

/// Compute the number of bytes needed to store `n_elements` values of the
/// given tensor type.
pub fn tensor_byte_size(dtype: GgufTensorType, n_elements: u64) -> u64 {
    let bs = dtype.block_size() as u64;
    let ts = dtype.type_size() as u64;
    // Number of blocks = ceil(n_elements / block_size)
    // Total bytes = n_blocks * type_size
    ((n_elements + bs - 1) / bs) * ts
}

// ---------------------------------------------------------------------------
// Helpers to reinterpret raw bytes as block slices
// ---------------------------------------------------------------------------

/// Interpret a byte slice as a slice of `BlockQ8_0` blocks.
///
/// Returns an error if the byte length is not a multiple of the block size.
pub fn bytes_as_q8_0_blocks(data: &[u8]) -> Result<&[BlockQ8_0], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ8_0>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q8_0 data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    // SAFETY: BlockQ8_0 is repr(C, packed) with no padding. We have verified
    // the data length is a multiple of the block size. The block contains only
    // u16 and i8 which have no alignment requirements with packed repr.
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ8_0, n_blocks) };
    Ok(blocks)
}

/// Interpret a byte slice as a slice of `BlockQ4_0` blocks.
///
/// Returns an error if the byte length is not a multiple of the block size.
pub fn bytes_as_q4_0_blocks(data: &[u8]) -> Result<&[BlockQ4_0], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ4_0>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q4_0 data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    // SAFETY: Same rationale as bytes_as_q8_0_blocks.
    let blocks =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_0, n_blocks) };
    Ok(blocks)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- f16 conversion tests --

    #[test]
    fn test_f16_to_f32_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_negative_zero() {
        assert_eq!(f16_to_f32(0x8000), -0.0);
        // Negative zero should compare equal to positive zero
        assert_eq!(f16_to_f32(0x8000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // f16 1.0 = 0 01111 0000000000 = 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0);
    }

    #[test]
    fn test_f16_to_f32_negative_one() {
        // f16 -1.0 = 1 01111 0000000000 = 0xBC00
        assert_eq!(f16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // f16 0.5 = 0 01110 0000000000 = 0x3800
        assert_eq!(f16_to_f32(0x3800), 0.5);
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        // f16 +inf = 0 11111 0000000000 = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
    }

    #[test]
    fn test_f16_to_f32_nan() {
        // f16 NaN = 0 11111 0000000001 = 0x7C01
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Smallest positive subnormal: 0 00000 0000000001 = 0x0001
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        // Should be approximately 2^-24 = 5.96e-8
        assert!((val - 5.960464e-8).abs() < 1e-12);
    }

    #[test]
    fn test_f16_roundtrip() {
        let test_values = [0.0f32, 1.0, -1.0, 0.5, 2.0, 100.0, -0.125];
        for &v in &test_values {
            let bits = f32_to_f16(v);
            let recovered = f16_to_f32(bits);
            assert!(
                (recovered - v).abs() < 0.01,
                "roundtrip failed for {}: got {}",
                v,
                recovered
            );
        }
    }

    // -- GgufTensorType tests --

    #[test]
    fn test_tensor_type_from_u32() {
        assert_eq!(GgufTensorType::from_u32(0).unwrap(), GgufTensorType::F32);
        assert_eq!(GgufTensorType::from_u32(1).unwrap(), GgufTensorType::F16);
        assert_eq!(GgufTensorType::from_u32(2).unwrap(), GgufTensorType::Q4_0);
        assert_eq!(GgufTensorType::from_u32(8).unwrap(), GgufTensorType::Q8_0);
        assert!(GgufTensorType::from_u32(4).is_err()); // Q4_2 removed
        assert!(GgufTensorType::from_u32(5).is_err()); // Q4_3 removed
        assert!(GgufTensorType::from_u32(999).is_err());
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(GgufTensorType::F32.block_size(), 1);
        assert_eq!(GgufTensorType::F16.block_size(), 1);
        assert_eq!(GgufTensorType::Q4_0.block_size(), 32);
        assert_eq!(GgufTensorType::Q8_0.block_size(), 32);
    }

    #[test]
    fn test_type_sizes() {
        assert_eq!(GgufTensorType::F32.type_size(), 4);
        assert_eq!(GgufTensorType::F16.type_size(), 2);
        assert_eq!(GgufTensorType::Q4_0.type_size(), 18);
        assert_eq!(GgufTensorType::Q8_0.type_size(), 34);
    }

    #[test]
    fn test_block_q8_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
    }

    #[test]
    fn test_block_q4_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
    }

    // -- Q8_0 dequantization tests --

    #[test]
    fn test_dequantize_q8_0_zeros() {
        let block = BlockQ8_0 {
            d: f32_to_f16(1.0),
            qs: [0; 32],
        };
        let result = dequantize_q8_0(&[block]);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantize_q8_0_ones() {
        let block = BlockQ8_0 {
            d: f32_to_f16(1.0),
            qs: [1; 32],
        };
        let result = dequantize_q8_0(&[block]);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q8_0_scale() {
        let block = BlockQ8_0 {
            d: f32_to_f16(0.5),
            qs: {
                let mut qs = [0i8; 32];
                qs[0] = 2;
                qs[1] = -4;
                qs[31] = 127;
                qs
            },
        };
        let result = dequantize_q8_0(&[block]);
        assert_eq!(result.len(), 32);
        // 0.5 * 2 = 1.0
        assert!((result[0] - 1.0).abs() < 1e-3);
        // 0.5 * -4 = -2.0
        assert!((result[1] - (-2.0)).abs() < 1e-3);
        // 0.5 * 127 = 63.5
        assert!((result[31] - 63.5).abs() < 0.1);
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        let blocks = vec![
            BlockQ8_0 {
                d: f32_to_f16(1.0),
                qs: [1; 32],
            },
            BlockQ8_0 {
                d: f32_to_f16(2.0),
                qs: [1; 32],
            },
        ];
        let result = dequantize_q8_0(&blocks);
        assert_eq!(result.len(), 64);
        // First block: 1.0 * 1 = ~1.0
        assert!((result[0] - 1.0).abs() < 1e-3);
        // Second block: 2.0 * 1 = ~2.0
        assert!((result[32] - 2.0).abs() < 1e-2);
    }

    // -- Q4_0 dequantization tests --

    #[test]
    fn test_dequantize_q4_0_zeros() {
        // Nibble value 8 maps to (8 - 8) = 0
        let block = BlockQ4_0 {
            d: f32_to_f16(1.0),
            qs: [0x88; 16], // both nibbles = 8
        };
        let result = dequantize_q4_0(&[block]);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequantize_q4_0_known_values() {
        // Low nibble = 0x0F = 15, high nibble = 0x00 = 0
        // For byte 0xF0: low = 0, high = 15
        // Wait, let's be precise: byte 0x0F means low=15, high=0
        // dequant: (15 - 8) * d = 7 * d for low, (0 - 8) * d = -8 * d for high
        let block = BlockQ4_0 {
            d: f32_to_f16(1.0),
            qs: {
                let mut qs = [0x88u8; 16]; // neutral = 0
                qs[0] = 0x0F; // low=15, high=0
                qs
            },
        };
        let result = dequantize_q4_0(&[block]);
        assert_eq!(result.len(), 32);
        // result[0] = (15 - 8) * 1.0 = 7.0
        assert!(
            (result[0] - 7.0).abs() < 1e-3,
            "expected 7.0, got {}",
            result[0]
        );
        // result[16] = (0 - 8) * 1.0 = -8.0
        assert!(
            (result[16] - (-8.0)).abs() < 1e-3,
            "expected -8.0, got {}",
            result[16]
        );
    }

    #[test]
    fn test_dequantize_q4_0_scale() {
        let block = BlockQ4_0 {
            d: f32_to_f16(0.5),
            qs: {
                let mut qs = [0x88u8; 16]; // neutral
                qs[0] = 0xAA; // low=0xA=10, high=0xA=10 -> (10-8)*0.5 = 1.0
                qs
            },
        };
        let result = dequantize_q4_0(&[block]);
        // (10 - 8) * 0.5 = 1.0
        assert!(
            (result[0] - 1.0).abs() < 1e-3,
            "expected 1.0, got {}",
            result[0]
        );
        assert!(
            (result[16] - 1.0).abs() < 1e-3,
            "expected 1.0, got {}",
            result[16]
        );
    }

    #[test]
    fn test_dequantize_q4_0_multiple_blocks() {
        let blocks = vec![
            BlockQ4_0 {
                d: f32_to_f16(1.0),
                qs: [0x88; 16],
            },
            BlockQ4_0 {
                d: f32_to_f16(2.0),
                qs: [0x88; 16],
            },
        ];
        let result = dequantize_q4_0(&blocks);
        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|&v| v.abs() < 1e-3));
    }

    // -- bytes_as_*_blocks tests --

    #[test]
    fn test_bytes_as_q8_0_blocks() {
        let data = vec![0u8; 34 * 3];
        let blocks = bytes_as_q8_0_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_bytes_as_q8_0_blocks_invalid_size() {
        let data = vec![0u8; 35]; // not a multiple of 34
        assert!(bytes_as_q8_0_blocks(&data).is_err());
    }

    #[test]
    fn test_bytes_as_q4_0_blocks() {
        let data = vec![0u8; 18 * 2];
        let blocks = bytes_as_q4_0_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_bytes_as_q4_0_blocks_invalid_size() {
        let data = vec![0u8; 19]; // not a multiple of 18
        assert!(bytes_as_q4_0_blocks(&data).is_err());
    }

    // -- tensor_byte_size tests --

    #[test]
    fn test_tensor_byte_size_f32() {
        assert_eq!(tensor_byte_size(GgufTensorType::F32, 100), 400);
    }

    #[test]
    fn test_tensor_byte_size_f16() {
        assert_eq!(tensor_byte_size(GgufTensorType::F16, 100), 200);
    }

    #[test]
    fn test_tensor_byte_size_q8_0() {
        // 100 elements / 32 per block = 3 blocks (we need n_elements divisible by block_size)
        assert_eq!(tensor_byte_size(GgufTensorType::Q8_0, 32), 34);
        assert_eq!(tensor_byte_size(GgufTensorType::Q8_0, 64), 68);
        assert_eq!(tensor_byte_size(GgufTensorType::Q8_0, 1024), 1088);
    }

    #[test]
    fn test_tensor_byte_size_q4_0() {
        assert_eq!(tensor_byte_size(GgufTensorType::Q4_0, 32), 18);
        assert_eq!(tensor_byte_size(GgufTensorType::Q4_0, 64), 36);
    }

    // ====================================================================
    // NEW TESTS: Quantization precision and edge cases
    // ====================================================================

    // -- Q8_0 precision with specific bit patterns --

    #[test]
    fn test_dequantize_q8_0_specific_bit_pattern() {
        // Scale = 0.25 as f16 (exact: 0x3400 = 0 01101 0000000000)
        // Verify the exact f16 bit pattern for 0.25
        let d_bits = f32_to_f16(0.25);
        assert_eq!(d_bits, 0x3400, "f16 bit pattern for 0.25 should be 0x3400");

        let mut qs = [0i8; 32];
        qs[0] = 127;   // max positive
        qs[1] = -128;  // min negative (note: i8 range is -128..=127)
        qs[2] = 0;     // zero
        qs[3] = 1;     // unit
        qs[4] = -1;    // negative unit

        let block = BlockQ8_0 { d: d_bits, qs };
        let result = dequantize_q8_0(&[block]);

        // 0.25 * 127 = 31.75
        assert!((result[0] - 31.75).abs() < 1e-3, "max positive: got {}", result[0]);
        // 0.25 * -128 = -32.0
        assert!((result[1] - (-32.0)).abs() < 1e-3, "min negative: got {}", result[1]);
        // 0.25 * 0 = 0.0
        assert_eq!(result[2], 0.0, "zero: got {}", result[2]);
        // 0.25 * 1 = 0.25
        assert!((result[3] - 0.25).abs() < 1e-3, "unit: got {}", result[3]);
        // 0.25 * -1 = -0.25
        assert!((result[4] - (-0.25)).abs() < 1e-3, "neg unit: got {}", result[4]);
    }

    #[test]
    fn test_dequantize_q8_0_zero_scale() {
        // When scale is zero, all dequantized values should be zero regardless of qs.
        let block = BlockQ8_0 {
            d: f32_to_f16(0.0),
            qs: [127; 32],
        };
        let result = dequantize_q8_0(&[block]);
        assert!(result.iter().all(|&v| v == 0.0), "zero scale should produce all zeros");
    }

    #[test]
    fn test_dequantize_q8_0_negative_scale() {
        // Negative scale: d=-1.0, qs=[1]... => output should be -1.0 for each
        let block = BlockQ8_0 {
            d: f32_to_f16(-1.0),
            qs: [1; 32],
        };
        let result = dequantize_q8_0(&[block]);
        for &v in &result {
            assert!((v - (-1.0)).abs() < 1e-3, "negative scale: got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q8_0_empty_blocks() {
        // Empty slice should produce empty output.
        let result = dequantize_q8_0(&[]);
        assert!(result.is_empty());
    }

    // -- Q4_0 precision with specific bit patterns --

    #[test]
    fn test_dequantize_q4_0_all_nibble_values() {
        // Test all 16 possible nibble values (0-15) in a single block.
        // Put nibbles 0..15 in the low positions and 0..15 in the high positions.
        let d_bits = f32_to_f16(1.0);
        let mut qs = [0u8; 16];
        for j in 0..16usize {
            // low nibble = j, high nibble = j
            qs[j] = (j as u8) | ((j as u8) << 4);
        }
        let block = BlockQ4_0 { d: d_bits, qs };
        let result = dequantize_q4_0(&[block]);

        // Low half (indices 0..16): (j - 8) * 1.0
        for j in 0..16 {
            let expected = (j as f32) - 8.0;
            assert!(
                (result[j] - expected).abs() < 1e-3,
                "low nibble {}: expected {}, got {}",
                j, expected, result[j]
            );
        }
        // High half (indices 16..32): same as low because both nibbles equal
        for j in 0..16 {
            let expected = (j as f32) - 8.0;
            assert!(
                (result[16 + j] - expected).abs() < 1e-3,
                "high nibble {}: expected {}, got {}",
                j, expected, result[16 + j]
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_zero_scale() {
        let block = BlockQ4_0 {
            d: f32_to_f16(0.0),
            qs: [0xFF; 16], // all nibbles = 15
        };
        let result = dequantize_q4_0(&[block]);
        assert!(result.iter().all(|&v| v == 0.0), "zero scale should produce all zeros");
    }

    #[test]
    fn test_dequantize_q4_0_empty_blocks() {
        let result = dequantize_q4_0(&[]);
        assert!(result.is_empty());
    }

    // -- f16 conversion: additional precision tests --

    #[test]
    fn test_f16_to_f32_specific_values() {
        // 2.0 = 0 10000 0000000000 = 0x4000
        assert_eq!(f16_to_f32(0x4000), 2.0);
        // -2.0 = 1 10000 0000000000 = 0xC000
        assert_eq!(f16_to_f32(0xC000), -2.0);
        // 0.1 (nearest f16 representable value)
        let bits = f32_to_f16(0.1);
        let recovered = f16_to_f32(bits);
        assert!((recovered - 0.1).abs() < 0.001, "0.1 round-trip: {}", recovered);
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // f16 -inf = 1 11111 0000000000 = 0xFC00
        let val = f16_to_f32(0xFC00);
        assert!(val.is_infinite());
        assert!(val < 0.0);
    }

    #[test]
    fn test_f16_max_value() {
        // f16 max = 0 11110 1111111111 = 0x7BFF = 65504.0
        let val = f16_to_f32(0x7BFF);
        assert!((val - 65504.0).abs() < 1.0, "f16 max should be 65504.0, got {}", val);
    }

    #[test]
    fn test_f32_to_f16_overflow_saturates_to_inf() {
        // Values larger than f16 max (65504) should saturate to infinity.
        let bits = f32_to_f16(100000.0);
        let val = f16_to_f32(bits);
        assert!(val.is_infinite(), "values > 65504 should become inf, got {}", val);
    }

    // -- GgufTensorType edge cases --

    #[test]
    fn test_tensor_type_removed_ids_4_5() {
        // Type IDs 4 and 5 were Q4_2 and Q4_3, now removed.
        assert!(GgufTensorType::from_u32(4).is_err());
        assert!(GgufTensorType::from_u32(5).is_err());
    }

    #[test]
    fn test_tensor_type_display() {
        assert_eq!(format!("{}", GgufTensorType::F32), "F32");
        assert_eq!(format!("{}", GgufTensorType::Q8_0), "Q8_0");
        assert_eq!(format!("{}", GgufTensorType::BF16), "BF16");
    }

    #[test]
    fn test_tensor_type_all_valid_ids() {
        // All valid IDs should parse without error (comprehensive coverage).
        let valid_ids = [
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        ];
        for &id in &valid_ids {
            assert!(
                GgufTensorType::from_u32(id).is_ok(),
                "type id {} should be valid",
                id
            );
        }
    }

    #[test]
    fn test_tensor_type_block_and_type_sizes_consistency() {
        // For non-quantized types, block_size should be 1 and type_size should be
        // the element size.
        assert_eq!(GgufTensorType::F64.block_size(), 1);
        assert_eq!(GgufTensorType::F64.type_size(), 8);
        assert_eq!(GgufTensorType::I8.block_size(), 1);
        assert_eq!(GgufTensorType::I8.type_size(), 1);
        assert_eq!(GgufTensorType::I16.block_size(), 1);
        assert_eq!(GgufTensorType::I16.type_size(), 2);
        assert_eq!(GgufTensorType::I32.block_size(), 1);
        assert_eq!(GgufTensorType::I32.type_size(), 4);
        assert_eq!(GgufTensorType::I64.block_size(), 1);
        assert_eq!(GgufTensorType::I64.type_size(), 8);
        assert_eq!(GgufTensorType::BF16.block_size(), 1);
        assert_eq!(GgufTensorType::BF16.type_size(), 2);
    }

    #[test]
    fn test_tensor_byte_size_zero_elements() {
        // Zero elements should produce zero bytes.
        assert_eq!(tensor_byte_size(GgufTensorType::F32, 0), 0);
        assert_eq!(tensor_byte_size(GgufTensorType::Q8_0, 0), 0);
    }

    #[test]
    fn test_tensor_byte_size_large() {
        // A 768x768 Q8_0 weight matrix
        let n_elements = 768u64 * 768;
        let expected = (n_elements / 32) * 34;
        assert_eq!(tensor_byte_size(GgufTensorType::Q8_0, n_elements), expected);
    }

    #[test]
    fn test_bytes_as_q8_0_blocks_empty() {
        let data = vec![];
        let blocks = bytes_as_q8_0_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_bytes_as_q4_0_blocks_empty() {
        let data = vec![];
        let blocks = bytes_as_q4_0_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 0);
    }

    // -- Cross-validate Q8_0 dequantization via both paths --

    #[test]
    fn test_q8_0_dequant_via_bytes_matches_direct() {
        // Create a block with known values, dequantize via bytes_as_q8_0_blocks
        // and via direct BlockQ8_0 construction; results must match.
        let scale = f32_to_f16(0.125);
        let mut raw = Vec::new();
        raw.extend_from_slice(&scale.to_le_bytes());
        for i in 0..32 {
            raw.push(i as u8); // i8 values 0..31
        }

        let blocks = bytes_as_q8_0_blocks(&raw).unwrap();
        let result_from_bytes = dequantize_q8_0(blocks);

        let block_direct = BlockQ8_0 {
            d: scale,
            qs: {
                let mut qs = [0i8; 32];
                for i in 0..32 {
                    qs[i] = i as i8;
                }
                qs
            },
        };
        let result_direct = dequantize_q8_0(&[block_direct]);

        assert_eq!(result_from_bytes.len(), result_direct.len());
        for i in 0..32 {
            assert!(
                (result_from_bytes[i] - result_direct[i]).abs() < 1e-6,
                "mismatch at {}: {} vs {}",
                i, result_from_bytes[i], result_direct[i]
            );
        }
    }
}
