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
            Self::IQ2XXS
            | Self::IQ2XS
            | Self::IQ3XXS
            | Self::IQ1S
            | Self::IQ4NL
            | Self::IQ3S
            | Self::IQ2S
            | Self::IQ4XS
            | Self::IQ1M => 256,
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
            Self::Q4_1 => 20,                               // 2*f16 + 16 bytes
            Self::Q5_0 => 22,                               // f16 + 4 + 16 bytes
            Self::Q5_1 => 24,                               // 2*f16 + 4 + 16 bytes
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(), // 34
            Self::Q8_1 => 36,                               // 2*f16 + 32 bytes
            // K-quant sizes (QK_K = 256)
            Self::Q2K => 2 * 2 + 256 / 16 + 256 / 4, // 84
            Self::Q3K => 2 + 256 / 4 + 256 / 8 + 12, // 110
            Self::Q4K => 2 * 2 + 12 + 256 / 2,       // 144
            Self::Q5K => 2 * 2 + 12 + 256 / 8 + 256 / 2, // 176
            Self::Q6K => 2 + 256 / 16 + 3 * 256 / 4, // 210
            Self::Q8K => 4 + 256 + 256 / 16 * 2,     // 292
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
// Constants
// ---------------------------------------------------------------------------

/// Number of elements per K-quant super-block.
pub const QK_K: usize = 256;

/// Size of the packed scale/min array in Q4_K and Q5_K blocks.
pub const K_SCALE_SIZE: usize = 12;

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
// Q4_1 quantization block: 20 bytes per block of 32 values
// ---------------------------------------------------------------------------

/// Q4_1 block: 4-bit quantization with f16 scale (d) and f16 minimum (m).
///
/// Layout: `d: f16 (2 bytes) | m: f16 (2 bytes) | qs: [u8; 16] (16 bytes)` = 20 bytes.
/// Dequantization: `y[j] = (low_nibble) * d + m`
///                 `y[j+16] = (high_nibble) * d + m`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_1 {
    pub d: u16,
    pub m: u16,
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

/// Number of elements per Q4_1 block.
pub const QK4_1: usize = 32;

// ---------------------------------------------------------------------------
// Q5_0 quantization block: 22 bytes per block of 32 values
// ---------------------------------------------------------------------------

/// Q5_0 block: 5-bit quantization with f16 scale (d), no minimum.
///
/// Layout: `d: f16 (2 bytes) | qh: [u8; 4] (4 bytes) | qs: [u8; 16] (16 bytes)` = 22 bytes.
/// The 5th bit for each value is stored in qh (as a 32-bit mask).
/// Dequantization: `y[j] = ((nibble | high_bit) - 16) * d`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_0 {
    pub d: u16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

/// Number of elements per Q5_0 block.
pub const QK5_0: usize = 32;

// ---------------------------------------------------------------------------
// Q5_1 quantization block: 24 bytes per block of 32 values
// ---------------------------------------------------------------------------

/// Q5_1 block: 5-bit quantization with f16 scale (d) and f16 minimum (m).
///
/// Layout: `d: f16 (2 bytes) | m: f16 (2 bytes) | qh: [u8; 4] (4 bytes) | qs: [u8; 16] (16 bytes)` = 24 bytes.
/// Dequantization: `y[j] = (nibble | high_bit) * d + m`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5_1 {
    pub d: u16,
    pub m: u16,
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

/// Number of elements per Q5_1 block.
pub const QK5_1: usize = 32;

// ---------------------------------------------------------------------------
// Q4_K quantization block: 144 bytes per super-block of 256 values
// ---------------------------------------------------------------------------

/// Q4_K block: 4-bit K-quant with 6-bit packed scales and mins.
///
/// 8 sub-blocks of 32 values each. Scales and mins are quantized with 6 bits
/// and packed into 12 bytes.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);

// ---------------------------------------------------------------------------
// Q5_K quantization block: 176 bytes per super-block of 256 values
// ---------------------------------------------------------------------------

/// Q5_K block: 5-bit K-quant with 6-bit packed scales and mins.
///
/// Same scale structure as Q4_K, but with an extra qh array for the 5th bit.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ5K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

const _: () = assert!(std::mem::size_of::<BlockQ5K>() == 176);

// ---------------------------------------------------------------------------
// Q6_K quantization block: 210 bytes per super-block of 256 values
// ---------------------------------------------------------------------------

/// Q6_K block: 6-bit K-quant with 8-bit signed scales, no mins.
///
/// 16 sub-blocks of 16 values each. Each value is 6 bits: 4 bits in ql,
/// 2 bits in qh. Scales are direct 8-bit signed integers.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: u16,
}

const _: () = assert!(std::mem::size_of::<BlockQ6K>() == 210);

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

// ---------------------------------------------------------------------------
// K-quant scale extraction helper
// ---------------------------------------------------------------------------

/// Extract 6-bit scale and min values from the packed Q4_K / Q5_K scales array.
///
/// Ported from llama.cpp's `get_scale_min_k4()`. The 12-byte scales array packs
/// 8 pairs of (scale, min) values using 6 bits each.
#[inline]
pub fn get_scale_min_k4(j: usize, scales: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

// ---------------------------------------------------------------------------
// Non-K dequantization functions
// ---------------------------------------------------------------------------

/// Dequantize a slice of Q4_1 blocks into f32 values.
///
/// Like Q4_0 but unsigned with a minimum offset: `y = nibble * d + m`
pub fn dequantize_q4_1(blocks: &[BlockQ4_1]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK4_1);
    for block in blocks {
        let d = f16_to_f32(block.d);
        let m = f16_to_f32(block.m);
        let mut tmp = [0.0f32; QK4_1];
        for j in 0..QK4_1 / 2 {
            let x0 = (block.qs[j] & 0x0F) as f32;
            let x1 = (block.qs[j] >> 4) as f32;
            tmp[j] = x0 * d + m;
            tmp[j + QK4_1 / 2] = x1 * d + m;
        }
        output.extend_from_slice(&tmp);
    }
    output
}

/// Dequantize a slice of Q5_0 blocks into f32 values.
///
/// 5-bit signed quantization: `y = ((nibble | high_bit) - 16) * d`
pub fn dequantize_q5_0(blocks: &[BlockQ5_0]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK5_0);
    for block in blocks {
        let d = f16_to_f32(block.d);
        let qh = u32::from_le_bytes(block.qh);
        let mut tmp = [0.0f32; QK5_0];
        for j in 0..QK5_0 / 2 {
            let xh_0 = ((qh >> (j as u32)) << 4) & 0x10;
            let xh_1 = (qh >> (j as u32 + 12)) & 0x10;
            let x0 = ((block.qs[j] & 0x0F) as u32 | xh_0) as i32 - 16;
            let x1 = ((block.qs[j] >> 4) as u32 | xh_1) as i32 - 16;
            tmp[j] = x0 as f32 * d;
            tmp[j + QK5_0 / 2] = x1 as f32 * d;
        }
        output.extend_from_slice(&tmp);
    }
    output
}

/// Dequantize a slice of Q5_1 blocks into f32 values.
///
/// 5-bit unsigned with minimum: `y = (nibble | high_bit) * d + m`
pub fn dequantize_q5_1(blocks: &[BlockQ5_1]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK5_1);
    for block in blocks {
        let d = f16_to_f32(block.d);
        let m = f16_to_f32(block.m);
        let qh = u32::from_le_bytes(block.qh);
        let mut tmp = [0.0f32; QK5_1];
        for j in 0..QK5_1 / 2 {
            let xh_0 = ((qh >> (j as u32)) << 4) & 0x10;
            let xh_1 = (qh >> (j as u32 + 12)) & 0x10;
            let x0 = (block.qs[j] & 0x0F) as u32 | xh_0;
            let x1 = (block.qs[j] >> 4) as u32 | xh_1;
            tmp[j] = x0 as f32 * d + m;
            tmp[j + QK5_1 / 2] = x1 as f32 * d + m;
        }
        output.extend_from_slice(&tmp);
    }
    output
}

// ---------------------------------------------------------------------------
// K-quant dequantization functions
// ---------------------------------------------------------------------------

/// Dequantize a slice of Q4_K blocks into f32 values.
///
/// 8 sub-blocks of 32 values each, 4-bit quants with 6-bit packed scales/mins.
/// `y = d * scale * (q & 0xF) - dmin * min_val`
pub fn dequantize_q4_k(blocks: &[BlockQ4K]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK_K);
    for block in blocks {
        let d = f16_to_f32(block.d);
        let dmin = f16_to_f32(block.dmin);
        let q = &block.qs;

        let mut is = 0usize;
        let mut q_offset = 0usize;
        for _j in 0..4 {
            // 64 values per iteration (two sub-blocks of 32)
            let (sc1, m1) = get_scale_min_k4(is, &block.scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                output.push(d1 * (q[q_offset + l] & 0xF) as f32 - m1);
            }
            for l in 0..32 {
                output.push(d2 * (q[q_offset + l] >> 4) as f32 - m2);
            }
            q_offset += 32;
            is += 2;
        }
    }
    output
}

/// Dequantize a slice of Q5_K blocks into f32 values.
///
/// Same scale structure as Q4_K but 5-bit quants (4 bits in qs + 1 bit in qh).
pub fn dequantize_q5_k(blocks: &[BlockQ5K]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK_K);
    for block in blocks {
        let d = f16_to_f32(block.d);
        let dmin = f16_to_f32(block.dmin);
        let ql = &block.qs;
        let qh = &block.qh;

        let mut is = 0usize;
        let mut ql_offset = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, &block.scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, &block.scales);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                let high = if qh[l] & u1 != 0 { 16 } else { 0 };
                output.push(d1 * ((ql[ql_offset + l] & 0xF) as f32 + high as f32) - m1);
            }
            for l in 0..32 {
                let high = if qh[l] & u2 != 0 { 16 } else { 0 };
                output.push(d2 * ((ql[ql_offset + l] >> 4) as f32 + high as f32) - m2);
            }
            ql_offset += 32;
            is += 2;
            u1 = u1.wrapping_shl(2);
            u2 = u2.wrapping_shl(2);
        }
    }
    output
}

/// Dequantize a slice of Q6_K blocks into f32 values.
///
/// 6-bit quants with 8-bit signed scales. No mins.
/// `y = d * scale * (q_6bit - 32)`
pub fn dequantize_q6_k(blocks: &[BlockQ6K]) -> Vec<f32> {
    let mut output = Vec::with_capacity(blocks.len() * QK_K);
    for block in blocks {
        let d = f16_to_f32(block.d);
        let ql = &block.ql;
        let qh = &block.qh;
        let sc = &block.scales;

        let mut buf = [0.0f32; QK_K];
        let mut ql_offset = 0usize;
        let mut qh_offset = 0usize;
        let mut sc_offset = 0usize;
        let mut out_offset = 0usize;

        for _n in 0..2 {
            // 128 values per iteration
            for l in 0..32 {
                let is = l / 16;
                let q1 =
                    ((ql[ql_offset + l] & 0xF) | (((qh[qh_offset + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 3) << 4))
                    as i32
                    - 32;
                let q3 =
                    ((ql[ql_offset + l] >> 4) | (((qh[qh_offset + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_offset + l + 32] >> 4) | (((qh[qh_offset + l] >> 6) & 3) << 4))
                    as i32
                    - 32;

                buf[out_offset + l] = d * sc[sc_offset + is] as f32 * q1 as f32;
                buf[out_offset + l + 32] = d * sc[sc_offset + is + 2] as f32 * q2 as f32;
                buf[out_offset + l + 64] = d * sc[sc_offset + is + 4] as f32 * q3 as f32;
                buf[out_offset + l + 96] = d * sc[sc_offset + is + 6] as f32 * q4 as f32;
            }
            ql_offset += 64;
            qh_offset += 32;
            sc_offset += 8;
            out_offset += 128;
        }
        output.extend_from_slice(&buf);
    }
    output
}

// ---------------------------------------------------------------------------
// Helpers to reinterpret raw bytes as new block slices
// ---------------------------------------------------------------------------

/// Interpret a byte slice as a slice of `BlockQ4_1` blocks.
pub fn bytes_as_q4_1_blocks(data: &[u8]) -> Result<&[BlockQ4_1], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ4_1>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q4_1 data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_1, n_blocks) };
    Ok(blocks)
}

/// Interpret a byte slice as a slice of `BlockQ5_0` blocks.
pub fn bytes_as_q5_0_blocks(data: &[u8]) -> Result<&[BlockQ5_0], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ5_0>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q5_0 data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ5_0, n_blocks) };
    Ok(blocks)
}

/// Interpret a byte slice as a slice of `BlockQ5_1` blocks.
pub fn bytes_as_q5_1_blocks(data: &[u8]) -> Result<&[BlockQ5_1], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ5_1>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q5_1 data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ5_1, n_blocks) };
    Ok(blocks)
}

/// Interpret a byte slice as a slice of `BlockQ4K` blocks.
pub fn bytes_as_q4_k_blocks(data: &[u8]) -> Result<&[BlockQ4K], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ4K>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q4_K data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4K, n_blocks) };
    Ok(blocks)
}

/// Interpret a byte slice as a slice of `BlockQ5K` blocks.
pub fn bytes_as_q5_k_blocks(data: &[u8]) -> Result<&[BlockQ5K], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ5K>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q5_K data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ5K, n_blocks) };
    Ok(blocks)
}

/// Interpret a byte slice as a slice of `BlockQ6K` blocks.
pub fn bytes_as_q6_k_blocks(data: &[u8]) -> Result<&[BlockQ6K], InferenceError> {
    let block_size = std::mem::size_of::<BlockQ6K>();
    if data.len() % block_size != 0 {
        return Err(InferenceError::GgufParse(format!(
            "Q6_K data length {} is not a multiple of block size {}",
            data.len(),
            block_size
        )));
    }
    let n_blocks = data.len() / block_size;
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ6K, n_blocks) };
    Ok(blocks)
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
    let blocks = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_0, n_blocks) };
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
        qs[0] = 127; // max positive
        qs[1] = -128; // min negative (note: i8 range is -128..=127)
        qs[2] = 0; // zero
        qs[3] = 1; // unit
        qs[4] = -1; // negative unit

        let block = BlockQ8_0 { d: d_bits, qs };
        let result = dequantize_q8_0(&[block]);

        // 0.25 * 127 = 31.75
        assert!(
            (result[0] - 31.75).abs() < 1e-3,
            "max positive: got {}",
            result[0]
        );
        // 0.25 * -128 = -32.0
        assert!(
            (result[1] - (-32.0)).abs() < 1e-3,
            "min negative: got {}",
            result[1]
        );
        // 0.25 * 0 = 0.0
        assert_eq!(result[2], 0.0, "zero: got {}", result[2]);
        // 0.25 * 1 = 0.25
        assert!((result[3] - 0.25).abs() < 1e-3, "unit: got {}", result[3]);
        // 0.25 * -1 = -0.25
        assert!(
            (result[4] - (-0.25)).abs() < 1e-3,
            "neg unit: got {}",
            result[4]
        );
    }

    #[test]
    fn test_dequantize_q8_0_zero_scale() {
        // When scale is zero, all dequantized values should be zero regardless of qs.
        let block = BlockQ8_0 {
            d: f32_to_f16(0.0),
            qs: [127; 32],
        };
        let result = dequantize_q8_0(&[block]);
        assert!(
            result.iter().all(|&v| v == 0.0),
            "zero scale should produce all zeros"
        );
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
                j,
                expected,
                result[j]
            );
        }
        // High half (indices 16..32): same as low because both nibbles equal
        for j in 0..16 {
            let expected = (j as f32) - 8.0;
            assert!(
                (result[16 + j] - expected).abs() < 1e-3,
                "high nibble {}: expected {}, got {}",
                j,
                expected,
                result[16 + j]
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
        assert!(
            result.iter().all(|&v| v == 0.0),
            "zero scale should produce all zeros"
        );
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
        assert!(
            (recovered - 0.1).abs() < 0.001,
            "0.1 round-trip: {}",
            recovered
        );
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
        assert!(
            (val - 65504.0).abs() < 1.0,
            "f16 max should be 65504.0, got {}",
            val
        );
    }

    #[test]
    fn test_f32_to_f16_overflow_saturates_to_inf() {
        // Values larger than f16 max (65504) should saturate to infinity.
        let bits = f32_to_f16(100000.0);
        let val = f16_to_f32(bits);
        assert!(
            val.is_infinite(),
            "values > 65504 should become inf, got {}",
            val
        );
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
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30,
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
                i,
                result_from_bytes[i],
                result_direct[i]
            );
        }
    }

    // ====================================================================
    // K-quant and non-K block struct size assertions
    // ====================================================================

    #[test]
    fn test_block_q4_1_size() {
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
    }

    #[test]
    fn test_block_q5_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), 22);
    }

    #[test]
    fn test_block_q5_1_size() {
        assert_eq!(std::mem::size_of::<BlockQ5_1>(), 24);
    }

    #[test]
    fn test_block_q4_k_size() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
    }

    #[test]
    fn test_block_q5_k_size() {
        assert_eq!(std::mem::size_of::<BlockQ5K>(), 176);
    }

    #[test]
    fn test_block_q6_k_size() {
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
    }

    // ====================================================================
    // get_scale_min_k4 tests
    // ====================================================================

    #[test]
    fn test_get_scale_min_k4_low_indices() {
        // For j < 4: scale = scales[j] & 63, min = scales[j+4] & 63
        let scales: [u8; 12] = [10, 20, 30, 40, 50, 60, 63, 0, 0, 0, 0, 0];
        let (sc, m) = get_scale_min_k4(0, &scales);
        assert_eq!(sc, 10);
        assert_eq!(m, 50);
        let (sc, m) = get_scale_min_k4(1, &scales);
        assert_eq!(sc, 20);
        assert_eq!(m, 60);
        let (sc, m) = get_scale_min_k4(3, &scales);
        assert_eq!(sc, 40);
        assert_eq!(m, 0);
    }

    #[test]
    fn test_get_scale_min_k4_high_indices() {
        // For j >= 4: more complex bit packing
        let mut scales = [0u8; 12];
        // Set up so we can test the high-index extraction.
        // j=4: sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
        //      m  = (scales[8] >> 4)   | ((scales[4] >> 6) << 4)
        scales[0] = 0b1100_0000; // >> 6 = 3, << 4 = 48
        scales[4] = 0b1000_0000; // >> 6 = 2, << 4 = 32
        scales[8] = 0b0101_0011; // & 0xF = 3, >> 4 = 5
        let (sc, m) = get_scale_min_k4(4, &scales);
        assert_eq!(sc, 3 | (3 << 4)); // 3 + 48 = 51
        assert_eq!(m, 5 | (2 << 4)); // 5 + 32 = 37
    }

    // ====================================================================
    // Q4_1 dequantization tests
    // ====================================================================

    #[test]
    fn test_dequantize_q4_1_basic() {
        // All nibbles = 0 => y = 0 * d + m = m for all elements
        let block = BlockQ4_1 {
            d: f32_to_f16(1.0),
            m: f32_to_f16(0.5),
            qs: [0; 16], // low=0, high=0
        };
        let result = dequantize_q4_1(&[block]);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 0.5).abs() < 1e-3, "expected 0.5, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q4_1_max_nibble() {
        // All nibbles = 15 => y = 15 * d + m
        let block = BlockQ4_1 {
            d: f32_to_f16(1.0),
            m: f32_to_f16(0.0),
            qs: [0xFF; 16],
        };
        let result = dequantize_q4_1(&[block]);
        for &v in &result {
            assert!((v - 15.0).abs() < 1e-3, "expected 15.0, got {}", v);
        }
    }

    // ====================================================================
    // Q5_0 dequantization tests
    // ====================================================================

    #[test]
    fn test_dequantize_q5_0_zero() {
        // With all qs = 0 and qh = 0, values = (0 | 0) - 16 = -16, y = -16 * d
        let block = BlockQ5_0 {
            d: f32_to_f16(1.0),
            qh: [0; 4],
            qs: [0; 16],
        };
        let result = dequantize_q5_0(&[block]);
        assert_eq!(result.len(), 32);
        // All nibbles = 0, no high bit => (0 - 16) * 1.0 = -16.0
        for &v in &result {
            assert!((v - (-16.0)).abs() < 1e-3, "expected -16.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q5_0_with_high_bits() {
        // nibble = 0, high_bit = 1 (for first element) => (0 | 16) - 16 = 0
        let block = BlockQ5_0 {
            d: f32_to_f16(1.0),
            qh: [0x01, 0, 0, 0], // bit 0 is set for element 0
            qs: [0; 16],
        };
        let result = dequantize_q5_0(&[block]);
        // Element 0: nibble=0, high_bit from qh bit 0 => xh_0 = ((1 >> 0) << 4) & 0x10 = 0x10
        // val = (0 | 0x10) - 16 = 0
        assert!(
            (result[0] - 0.0).abs() < 1e-3,
            "expected 0.0, got {}",
            result[0]
        );
    }

    // ====================================================================
    // Q5_1 dequantization tests
    // ====================================================================

    #[test]
    fn test_dequantize_q5_1_basic() {
        // All nibbles = 0, no high bits => y = 0 * d + m = m
        let block = BlockQ5_1 {
            d: f32_to_f16(1.0),
            m: f32_to_f16(2.0),
            qh: [0; 4],
            qs: [0; 16],
        };
        let result = dequantize_q5_1(&[block]);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert!((v - 2.0).abs() < 1e-3, "expected 2.0, got {}", v);
        }
    }

    // ====================================================================
    // Q4_K dequantization tests
    // ====================================================================

    #[test]
    fn test_dequantize_q4_k_zeros() {
        // All zeros except d=1.0 and dmin=0.0 => all output should be 0
        let block = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [0; 12],
            qs: [0; 128],
        };
        let result = dequantize_q4_k(&[block]);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q4_k_basic() {
        // Set d=1.0, dmin=0.0, scale[0]=1 (first 6 bits), all qs nibbles = 5
        // y = d * 1 * 5 - dmin * 0 = 5.0
        let mut block = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [0; 12],
            qs: [0x55; 128], // low=5, high=5
        };
        block.scales[0] = 1; // scale for sub-block 0 = 1
        block.scales[1] = 1; // scale for sub-block 1 = 1

        let result = dequantize_q4_k(&[block]);
        // First 32 elements use sub-block 0 (low nibbles): d*1*5 = 5.0
        for i in 0..32 {
            assert!(
                (result[i] - 5.0).abs() < 1e-3,
                "elem {}: expected 5.0, got {}",
                i,
                result[i]
            );
        }
    }

    // ====================================================================
    // Q6_K dequantization tests
    // ====================================================================

    #[test]
    fn test_dequantize_q6_k_zeros() {
        // With d=1.0, all ql/qh=0, scales=0 => output = d * 0 * (0 - 32) = 0
        let block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: f32_to_f16(1.0),
        };
        let result = dequantize_q6_k(&[block]);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert!(v.abs() < 1e-6, "expected 0.0, got {}", v);
        }
    }

    #[test]
    fn test_dequantize_q6_k_known_value() {
        // Set scale[0] = 1, d = 1.0
        // ql[0] = 0x21 (low nibble = 1), qh[0] = 0b00000001 (bits 0,1 = 01 for first element)
        // q1 = (1 | (1 << 4)) - 32 = (1 | 16) - 32 = 17 - 32 = -15
        // y = 1.0 * 1 * (-15) = -15.0
        let mut block = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: f32_to_f16(1.0),
        };
        block.scales[0] = 1;
        block.ql[0] = 0x01; // low nibble = 1
        block.qh[0] = 0x01; // bits 0,1 for element 0 = 01

        let result = dequantize_q6_k(&[block]);
        // q1 = (1 | (1 << 4)) - 32 = 17 - 32 = -15
        assert!(
            (result[0] - (-15.0)).abs() < 1e-3,
            "expected -15.0, got {}",
            result[0]
        );
    }

    // ====================================================================
    // bytes_as_*_blocks tests for new types
    // ====================================================================

    #[test]
    fn test_bytes_as_q4_1_blocks() {
        let data = vec![0u8; 20 * 3];
        let blocks = bytes_as_q4_1_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_bytes_as_q4_1_blocks_invalid() {
        assert!(bytes_as_q4_1_blocks(&vec![0u8; 21]).is_err());
    }

    #[test]
    fn test_bytes_as_q5_0_blocks() {
        let data = vec![0u8; 22 * 2];
        let blocks = bytes_as_q5_0_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_bytes_as_q5_1_blocks() {
        let data = vec![0u8; 24 * 2];
        let blocks = bytes_as_q5_1_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_bytes_as_q4_k_blocks() {
        let data = vec![0u8; 144 * 2];
        let blocks = bytes_as_q4_k_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_bytes_as_q5_k_blocks() {
        let data = vec![0u8; 176];
        let blocks = bytes_as_q5_k_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_bytes_as_q6_k_blocks() {
        let data = vec![0u8; 210];
        let blocks = bytes_as_q6_k_blocks(&data).unwrap();
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_bytes_as_q6_k_blocks_invalid() {
        assert!(bytes_as_q6_k_blocks(&vec![0u8; 211]).is_err());
    }

    // ====================================================================
    // Empty blocks produce empty output
    // ====================================================================

    #[test]
    fn test_dequantize_empty_new_types() {
        assert!(dequantize_q4_1(&[]).is_empty());
        assert!(dequantize_q5_0(&[]).is_empty());
        assert!(dequantize_q5_1(&[]).is_empty());
        assert!(dequantize_q4_k(&[]).is_empty());
        assert!(dequantize_q5_k(&[]).is_empty());
        assert!(dequantize_q6_k(&[]).is_empty());
    }

    // ====================================================================
    // Output length correctness
    // ====================================================================

    #[test]
    fn test_dequantize_output_lengths() {
        // Non-K types: 32 elements per block
        let q4_1 = BlockQ4_1 {
            d: 0,
            m: 0,
            qs: [0; 16],
        };
        assert_eq!(dequantize_q4_1(&[q4_1, q4_1]).len(), 64);

        let q5_0 = BlockQ5_0 {
            d: 0,
            qh: [0; 4],
            qs: [0; 16],
        };
        assert_eq!(dequantize_q5_0(&[q5_0]).len(), 32);

        let q5_1 = BlockQ5_1 {
            d: 0,
            m: 0,
            qh: [0; 4],
            qs: [0; 16],
        };
        assert_eq!(dequantize_q5_1(&[q5_1]).len(), 32);

        // K-quant types: 256 elements per block
        let q4_k = BlockQ4K {
            d: 0,
            dmin: 0,
            scales: [0; 12],
            qs: [0; 128],
        };
        assert_eq!(dequantize_q4_k(&[q4_k]).len(), 256);

        let q5_k = BlockQ5K {
            d: 0,
            dmin: 0,
            scales: [0; 12],
            qh: [0; 32],
            qs: [0; 128],
        };
        assert_eq!(dequantize_q5_k(&[q5_k]).len(), 256);

        let q6_k = BlockQ6K {
            ql: [0; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: 0,
        };
        assert_eq!(dequantize_q6_k(&[q6_k]).len(), 256);
    }

    // ====================================================================
    // Deep dequantization tests with non-trivial data
    // ====================================================================

    #[test]
    fn test_dequantize_q4_1_varied_nibbles() {
        // d=0.5, m=1.0, qs: byte i has low=i%16, high=(15-i)%16
        let block = BlockQ4_1 {
            d: f32_to_f16(0.5),
            m: f32_to_f16(1.0),
            qs: {
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let lo = (j % 16) as u8;
                    let hi = ((15 - j) % 16) as u8;
                    qs[j] = lo | (hi << 4);
                }
                qs
            },
        };
        let result = dequantize_q4_1(&[block]);
        assert_eq!(result.len(), 32);
        // Verify a few specific values: y = nibble * d + m
        // Element 0: low nibble of qs[0] = 0 => 0 * 0.5 + 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 1e-2, "elem 0: got {}", result[0]);
        // Element 5: low nibble of qs[5] = 5 => 5 * 0.5 + 1.0 = 3.5
        assert!((result[5] - 3.5).abs() < 1e-2, "elem 5: got {}", result[5]);
        // Element 16: high nibble of qs[0] = 15 => 15 * 0.5 + 1.0 = 8.5
        assert!(
            (result[16] - 8.5).abs() < 1e-2,
            "elem 16: got {}",
            result[16]
        );
        // Element 31: high nibble of qs[15] = 0 => 0 * 0.5 + 1.0 = 1.0
        assert!(
            (result[31] - 1.0).abs() < 1e-2,
            "elem 31: got {}",
            result[31]
        );
    }

    #[test]
    fn test_dequantize_q5_0_all_high_bits_set() {
        // d=1.0, all qs nibbles = 0, all qh bits set
        // For element j (j<16): xh_0 = ((0xFFFFFFFF >> j) << 4) & 0x10 = 0x10
        //   val = (0 | 16) - 16 = 0 => y = 0.0
        let block = BlockQ5_0 {
            d: f32_to_f16(1.0),
            qh: [0xFF; 4], // all bits set
            qs: [0; 16],
        };
        let result = dequantize_q5_0(&[block]);
        assert_eq!(result.len(), 32);
        // All elements: (0 | 16) - 16 = 0 for both halves
        for i in 0..32 {
            assert!(
                (result[i] - 0.0).abs() < 1e-3,
                "elem {}: expected 0.0, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q5_0_mixed_bits() {
        // d=2.0, qs[0] = 0x31 (low=1, high=3), qh bit 0 set, bit 12 clear
        // Element 0: xh_0 = ((qh>>0)<<4) & 0x10 = 0x10; x0 = (1|16) - 16 = 1; y=2.0
        // Element 16: xh_1 = ((qh>>12)) & 0x10 = (qh>>12)&0x10; for qh=0x01: (1>>12)=0, so 0
        //   x1 = (3|0) - 16 = -13; y=-26.0
        let block = BlockQ5_0 {
            d: f32_to_f16(2.0),
            qh: [0x01, 0, 0, 0], // only bit 0 set
            qs: {
                let mut qs = [0x88u8; 16]; // nibbles = 8 => (8|0)-16=-8 or (8|16)-16=8
                qs[0] = 0x31; // low=1, high=3
                qs
            },
        };
        let result = dequantize_q5_0(&[block]);
        // Element 0: xh_0 = 16 (bit 0 set), x0 = (1|16)-16 = 1, y = 2.0
        assert!(
            (result[0] - 2.0).abs() < 1e-2,
            "elem 0: expected 2.0, got {}",
            result[0]
        );
        // Element 16: xh_1 = 0 (bit 12 not set), x1 = (3|0)-16 = -13, y = -26.0
        assert!(
            (result[16] - (-26.0)).abs() < 1e-2,
            "elem 16: expected -26.0, got {}",
            result[16]
        );
    }

    #[test]
    fn test_dequantize_q5_1_with_high_bits_and_min() {
        // d=0.5, m=2.0, qs[0] = 0xF0 (low=0, high=15)
        // For element j=0: xh_0 = ((qh >> 0) << 4) & 0x10 — needs qh bit 0
        // For element 16 (j=0 high half): xh_1 = ((qh >> 12)) & 0x10 — needs qh bit 16
        let block = BlockQ5_1 {
            d: f32_to_f16(0.5),
            m: f32_to_f16(2.0),
            qh: {
                let val: u32 = (1 << 0) | (1 << 16); // bit 0 for elem 0, bit 16 for elem 16
                val.to_le_bytes()
            },
            qs: {
                let mut qs = [0u8; 16];
                qs[0] = 0xF0; // low=0, high=15
                qs
            },
        };
        let result = dequantize_q5_1(&[block]);
        // Element 0: x0 = (0|16) = 16, y = 16*0.5 + 2.0 = 10.0
        assert!(
            (result[0] - 10.0).abs() < 1e-2,
            "elem 0: expected 10.0, got {}",
            result[0]
        );
        // Element 16: x1 = (15|16) = 31, y = 31*0.5 + 2.0 = 17.5
        assert!(
            (result[16] - 17.5).abs() < 1e-2,
            "elem 16: expected 17.5, got {}",
            result[16]
        );
    }

    #[test]
    fn test_dequantize_q4_k_with_mins_and_high_scale_indices() {
        // Exercise scale indices 4-7 (the high-index path in get_scale_min_k4)
        // d=1.0, dmin=0.5
        // Set scales so that sub-blocks 4-7 have non-trivial scale and min values
        // qs = 0x33 (low=3, high=3) throughout
        let mut block = BlockQ4K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.5),
            scales: [0; 12],
            qs: [0x33; 128], // all nibbles = 3
        };
        // Set low-index scales (0-3): scale=5, min=2
        for i in 0..4 {
            block.scales[i] = 5; // scale for sub-block i
            block.scales[i + 4] = 2; // min for sub-block i
        }
        // For high indices (4-7), we need the packed format:
        // scales[8] = (sc4 & 0xF) | (m4 & 0xF) << 4
        // sc4 = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
        // We want sc4=3, m4=1:
        // scales[8] = (3 & 0xF) | (1 << 4) = 3 | 16 = 19
        // scales[0] bits 6-7 must be 0 (already 5 = 0b00000101, ok)
        // scales[4] bits 6-7 for m4 must be 0 (already 2 = 0b00000010, ok)
        block.scales[8] = 19; // sc4=3, m4=1 (lower 4 bits of each)
        block.scales[9] = 19; // sc5=3, m5=1
        block.scales[10] = 19; // sc6=3, m6=1
        block.scales[11] = 19; // sc7=3, m7=1

        let result = dequantize_q4_k(&[block]);
        assert_eq!(result.len(), 256);

        // Sub-block 0 (elements 0-31, low nibbles of qs[0..32]):
        // y = d * scale * (q & 0xF) - dmin * min = 1.0 * 5 * 3 - 0.5 * 2 = 15.0 - 1.0 = 14.0
        for i in 0..32 {
            assert!(
                (result[i] - 14.0).abs() < 1e-2,
                "sub-block 0, elem {}: expected 14.0, got {}",
                i,
                result[i]
            );
        }

        // Sub-block 4 (elements 128-159):
        // sc4=3, m4=1, y = 1.0 * 3 * 3 - 0.5 * 1 = 9.0 - 0.5 = 8.5
        for i in 128..160 {
            assert!(
                (result[i] - 8.5).abs() < 1e-2,
                "sub-block 4, elem {}: expected 8.5, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q5_k_with_high_bits() {
        // Exercise the u1/u2 rotation through all 4 iterations (the wrapping_shl fix)
        // d=1.0, dmin=0.0. Set all 8 sub-block scales to 1 so we can verify values
        // across all iterations, including iteration 3 (the overflow iteration).
        let mut block = BlockQ5K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.0),
            scales: [0; 12],
            qh: [0; 32],
            qs: [0; 128], // all nibbles = 0
        };
        // Set low-index scales (sub-blocks 0-3): scale=1, min=0
        for i in 0..4 {
            block.scales[i] = 1;
        }
        // Set high-index scales (sub-blocks 4-7): sc=1, min=0
        // get_scale_min_k4(4): sc = (scales[8] & 0xF) | ((scales[0] >> 6) << 4)
        // We want sc=1: scales[8] & 0xF = 1, scales[0] bits 6-7 = 0 (already true since scales[0]=1)
        // min=0: (scales[8] >> 4) | ((scales[4] >> 6) << 4) = 0, so scales[8] >> 4 = 0
        // scales[8] = 1 (low 4 bits = 1, high 4 bits = 0)
        block.scales[8] = 1; // sub-block 4: sc=1, m=0
        block.scales[9] = 1; // sub-block 5: sc=1, m=0
        block.scales[10] = 1; // sub-block 6: sc=1, m=0
        block.scales[11] = 1; // sub-block 7: sc=1, m=0

        // Set qh[0] = 0xFF: bits 0-7 all set
        // Iteration 0: u1=1, u2=2 => qh[0]&1=1 (high=16), qh[0]&2=1 (high=16)
        // Iteration 1: u1=4, u2=8 => qh[0]&4=1 (high=16), qh[0]&8=1 (high=16)
        // Iteration 2: u1=16, u2=32 => qh[0]&16=1 (high=16), qh[0]&32=1 (high=16)
        // Iteration 3: u1=64, u2=128 => qh[0]&64=1 (high=16), qh[0]&128=1 (high=16)
        block.qh[0] = 0xFF;

        let result = dequantize_q5_k(&[block]);
        assert_eq!(result.len(), 256);

        // Element 0 (iteration 0, l=0): high=16, q=0+16=16, y=1.0*1*16=16.0
        assert!(
            (result[0] - 16.0).abs() < 1e-2,
            "elem 0: expected 16.0, got {}",
            result[0]
        );

        // Element 32 (iteration 0, second inner loop, l=0): high=16, y=16.0
        assert!(
            (result[32] - 16.0).abs() < 1e-2,
            "elem 32: expected 16.0, got {}",
            result[32]
        );

        // Element 64 (iteration 1, l=0): u1=4, qh[0]&4!=0 => high=16, y=16.0
        assert!(
            (result[64] - 16.0).abs() < 1e-2,
            "elem 64: expected 16.0, got {}",
            result[64]
        );

        // Element 192 (iteration 3, l=0): u1=64, qh[0]&64!=0 => high=16
        // This is the iteration that used to overflow (u1 <<= 2 on u8=64)
        // With wrapping_shl, it completes without panic. Sub-block 6 has sc=1.
        assert!(
            (result[192] - 16.0).abs() < 1e-2,
            "elem 192 (overflow iter): expected 16.0, got {}",
            result[192]
        );
    }

    #[test]
    fn test_dequantize_q5_k_no_high_bits() {
        // Same as above but qh=0, so no high bits
        let mut block = BlockQ5K {
            d: f32_to_f16(1.0),
            dmin: f32_to_f16(0.5),
            scales: [0; 12],
            qh: [0; 32],
            qs: [0x55; 128], // all nibbles = 5
        };
        block.scales[0] = 2; // scale for sub-block 0
        block.scales[4] = 1; // min for sub-block 0

        let result = dequantize_q5_k(&[block]);
        // Sub-block 0, first 32 elements:
        // q = (5 & 0xF) + 0 = 5 (no high bit)
        // y = 1.0 * 2 * 5 - 0.5 * 1 = 10.0 - 0.5 = 9.5
        for i in 0..32 {
            assert!(
                (result[i] - 9.5).abs() < 1e-2,
                "elem {}: expected 9.5, got {}",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q6_k_all_sub_blocks() {
        // Set scale[i]=i+1 for i=0..16, d=0.5
        // ql all = 0x21 (low nibble=1 for first half, =2 for second half)
        // qh all = 0, so q6 = ql nibble only
        let mut block = BlockQ6K {
            ql: [0x21; 128],
            qh: [0; 64],
            scales: [0; 16],
            d: f32_to_f16(0.5),
        };
        for i in 0..16 {
            block.scales[i] = (i + 1) as i8;
        }
        let result = dequantize_q6_k(&[block]);
        assert_eq!(result.len(), 256);

        // Sub-block 0 (first 16 elements): scale=1
        // q = (ql & 0xF) | ((qh & 3) << 4) = 1 | 0 = 1, q-32 = -31
        // y = 0.5 * 1 * (-31) = -15.5
        assert!(
            (result[0] - (-15.5)).abs() < 1e-2,
            "elem 0: expected -15.5, got {}",
            result[0]
        );

        // Sub-block 1 (elements 16-31): scale=2
        // Same ql pattern, q=1, y = 0.5 * 2 * (-31) = -31.0
        assert!(
            (result[16] - (-31.0)).abs() < 1e-2,
            "elem 16: expected -31.0, got {}",
            result[16]
        );
    }

    #[test]
    fn test_dequantize_q6_k_with_qh_bits() {
        // Verify that qh bits correctly contribute to the 6-bit value
        let mut block = BlockQ6K {
            ql: [0; 128], // all nibbles = 0
            qh: [0; 64],
            scales: [0; 16],
            d: f32_to_f16(1.0),
        };
        block.scales[0] = 1;
        // qh[0] bits 0,1 correspond to element 0's upper 2 bits
        // For the first 32 elements: qh_val = ((qh[j] >> 0) & 3) << 4
        block.qh[0] = 0x03; // bits 0,1 = 11 for element 0 => qh_val = 3<<4 = 48
                            // q = (0 | 48) - 32 = 16
                            // y = 1.0 * 1 * 16 = 16.0

        let result = dequantize_q6_k(&[block]);
        assert!(
            (result[0] - 16.0).abs() < 1e-2,
            "elem 0: expected 16.0, got {}",
            result[0]
        );
    }
}
