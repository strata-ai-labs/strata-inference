// M1: GGUF v3 binary parser — GgufFile, KV metadata, tensor info

pub mod quant;
pub mod tensor;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use tracing::{debug, info};

use crate::error::InferenceError;
use quant::GgufTensorType;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// GGUF magic number: ASCII "GGUF" as a little-endian u32.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Maximum GGUF version we support.
const GGUF_MAX_VERSION: u32 = 3;

/// Default alignment for the data section (bytes).
const GGUF_DEFAULT_ALIGNMENT: u32 = 32;

// ---------------------------------------------------------------------------
// GgufValue — typed metadata values
// ---------------------------------------------------------------------------

/// A typed value from a GGUF key-value pair.
///
/// GGUF supports 13 value types (IDs 0..12). Arrays contain homogeneous
/// elements stored as a nested `Vec<GgufValue>` (the inner values will all
/// be the same variant).
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Human-readable type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "U8",
            Self::I8(_) => "I8",
            Self::U16(_) => "U16",
            Self::I16(_) => "I16",
            Self::U32(_) => "U32",
            Self::I32(_) => "I32",
            Self::U64(_) => "U64",
            Self::I64(_) => "I64",
            Self::F32(_) => "F32",
            Self::F64(_) => "F64",
            Self::Bool(_) => "Bool",
            Self::String(_) => "String",
            Self::Array(_) => "Array",
        }
    }
}

// ---------------------------------------------------------------------------
// GgufValueType — KV value type IDs
// ---------------------------------------------------------------------------

/// GGUF KV value type IDs (matches gguf_type enum in gguf.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Result<Self, InferenceError> {
        match v {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(InferenceError::GgufParse(format!(
                "unknown GGUF value type: {}",
                v
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

/// Information about a single tensor stored in a GGUF file.
///
/// The `offset` field is relative to the start of the tensor data section,
/// not the start of the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight").
    pub name: String,
    /// Number of dimensions (1..=4 typically).
    pub n_dims: u32,
    /// Size along each dimension. Length == n_dims.
    pub dims: Vec<u64>,
    /// Data type (quantization format).
    pub dtype: GgufTensorType,
    /// Byte offset relative to the data section start.
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements in the tensor.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    /// Byte size of this tensor's data.
    pub fn byte_size(&self) -> u64 {
        quant::tensor_byte_size(self.dtype, self.n_elements())
    }
}

// ---------------------------------------------------------------------------
// GgufFile
// ---------------------------------------------------------------------------

/// A parsed GGUF file providing access to metadata and tensor data.
///
/// The file is memory-mapped so tensor data can be accessed zero-copy.
pub struct GgufFile {
    /// KV metadata from the file header.
    metadata: HashMap<String, GgufValue>,
    /// Information about every tensor in the file.
    tensor_infos: Vec<TensorInfo>,
    /// Byte offset where the tensor data section starts (absolute file offset).
    data_offset: u64,
    /// Alignment (in bytes) used for the data section.
    alignment: u32,
    /// Memory-mapped file contents (entire file).
    mmap: memmap2::Mmap,
}

impl GgufFile {
    // -- Construction -------------------------------------------------------

    /// Open and parse a GGUF file from the given path.
    ///
    /// This reads the header, all KV pairs, and all tensor info entries.
    /// The file is then memory-mapped so tensor data can be read zero-copy.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, InferenceError> {
        let path = path.as_ref();
        info!("opening GGUF file: {}", path.display());

        let file = File::open(path)?;

        // Memory-map the file for later zero-copy tensor access.
        // SAFETY: The file remains open for the lifetime of the Mmap. We do
        // not modify the file while it is mapped. External modifications are
        // undefined behaviour, but we accept this (same as llama.cpp).
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        // Use a BufReader over the raw file for sequential header parsing.
        let mut reader = BufReader::new(File::open(path)?);

        // ---- Magic ----
        let magic = read_u32(&mut reader)?;
        if magic != GGUF_MAGIC {
            return Err(InferenceError::InvalidMagic(magic));
        }

        // ---- Version ----
        let version = read_u32(&mut reader)?;
        if version == 0 || version > GGUF_MAX_VERSION {
            return Err(InferenceError::UnsupportedVersion(version));
        }
        if version == 1 {
            return Err(InferenceError::UnsupportedVersion(version));
        }
        info!("GGUF version {}", version);

        // ---- Tensor and KV counts ----
        // GGUF spec defines these as int64_t (signed). Read as u64 then
        // validate they are non-negative (high bit not set) and reasonable.
        let n_tensors_raw = read_u64(&mut reader)?;
        let n_kv_raw = read_u64(&mut reader)?;
        if n_tensors_raw > i64::MAX as u64 {
            return Err(InferenceError::GgufParse(format!(
                "n_tensors value {} is negative (interpreted as signed)",
                n_tensors_raw
            )));
        }
        if n_kv_raw > i64::MAX as u64 {
            return Err(InferenceError::GgufParse(format!(
                "n_kv value {} is negative (interpreted as signed)",
                n_kv_raw
            )));
        }
        const MAX_TENSORS: u64 = 1_000_000;
        const MAX_KV: u64 = 10_000_000;
        if n_tensors_raw > MAX_TENSORS {
            return Err(InferenceError::GgufParse(format!(
                "n_tensors {} exceeds maximum {}",
                n_tensors_raw, MAX_TENSORS
            )));
        }
        if n_kv_raw > MAX_KV {
            return Err(InferenceError::GgufParse(format!(
                "n_kv {} exceeds maximum {}",
                n_kv_raw, MAX_KV
            )));
        }
        let n_tensors = n_tensors_raw;
        let n_kv = n_kv_raw;
        info!("n_tensors: {}, n_kv: {}", n_tensors, n_kv);

        // ---- KV pairs ----
        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for i in 0..n_kv {
            let key = read_gguf_string(&mut reader)?;
            let value_type_id = read_u32(&mut reader)?;
            let value_type = GgufValueType::from_u32(value_type_id)?;
            let value = read_gguf_value(&mut reader, value_type)?;
            debug!("KV[{}]: {} = {:?}", i, key, value.type_name());
            metadata.insert(key, value);
        }

        // ---- Alignment ----
        let alignment = match metadata.get("general.alignment") {
            Some(GgufValue::U32(a)) => {
                let a = *a;
                if a == 0 || (a & (a - 1)) != 0 {
                    return Err(InferenceError::GgufParse(format!(
                        "alignment {} is not a power of 2",
                        a
                    )));
                }
                a
            }
            _ => GGUF_DEFAULT_ALIGNMENT,
        };
        debug!("alignment: {} bytes", alignment);

        // ---- Tensor info ----
        let mut tensor_infos = Vec::with_capacity(n_tensors as usize);
        for i in 0..n_tensors {
            let name = read_gguf_string(&mut reader)?;
            let n_dims = read_u32(&mut reader)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                // GGUF spec stores dims as int64 (signed), but they are always non-negative.
                let dim = read_u64(&mut reader)?;
                dims.push(dim);
            }
            let dtype_id = read_u32(&mut reader)?;
            let dtype = GgufTensorType::from_u32(dtype_id)?;
            let offset = read_u64(&mut reader)?;

            debug!(
                "tensor[{}]: {} shape={:?} dtype={} offset={}",
                i, name, dims, dtype, offset
            );
            tensor_infos.push(TensorInfo {
                name,
                n_dims,
                dims,
                dtype,
                offset,
            });
        }

        // ---- Data section offset ----
        // Current stream position is at the end of the tensor info section.
        // The data section starts at the next alignment boundary.
        let pos = reader.stream_position()?;
        let data_offset = align_offset(pos, alignment as u64);
        info!(
            "data section starts at offset {} (header ended at {})",
            data_offset, pos
        );

        Ok(GgufFile {
            metadata,
            tensor_infos,
            data_offset,
            alignment,
            mmap,
        })
    }

    // -- Metadata getters ---------------------------------------------------

    /// Get a raw metadata value by key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a string metadata value.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Get a u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get an i32 metadata value.
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        match self.metadata.get(key) {
            Some(GgufValue::I32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a u64 metadata value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(GgufValue::U64(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get an f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a bool metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(GgufValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a string array metadata value.
    pub fn get_str_array(&self, key: &str) -> Option<Vec<&str>> {
        match self.metadata.get(key) {
            Some(GgufValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        GgufValue::String(s) => result.push(s.as_str()),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get an f32 array metadata value.
    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key) {
            Some(GgufValue::Array(arr)) => {
                let mut result = Vec::with_capacity(arr.len());
                for item in arr {
                    match item {
                        GgufValue::F32(v) => result.push(*v),
                        _ => return None,
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }

    /// Get a required string value, returning an error if missing.
    pub fn require_str(&self, key: &str) -> Result<&str, InferenceError> {
        self.get_str(key)
            .ok_or_else(|| InferenceError::MissingKey(key.to_string()))
    }

    /// Get a required u32 value, returning an error if missing.
    pub fn require_u32(&self, key: &str) -> Result<u32, InferenceError> {
        self.get_u32(key)
            .ok_or_else(|| InferenceError::MissingKey(key.to_string()))
    }

    /// Get a required f32 value, returning an error if missing.
    pub fn require_f32(&self, key: &str) -> Result<f32, InferenceError> {
        self.get_f32(key)
            .ok_or_else(|| InferenceError::MissingKey(key.to_string()))
    }

    // -- Tensor info accessors ----------------------------------------------

    /// Get all tensor info entries.
    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.tensor_infos
    }

    /// Number of tensors in the file.
    pub fn n_tensors(&self) -> usize {
        self.tensor_infos.len()
    }

    /// Find a tensor by name, returning its index and info.
    pub fn find_tensor(&self, name: &str) -> Option<(usize, &TensorInfo)> {
        self.tensor_infos
            .iter()
            .enumerate()
            .find(|(_, t)| t.name == name)
    }

    /// Find a tensor by name, returning an error if not found.
    pub fn require_tensor(&self, name: &str) -> Result<&TensorInfo, InferenceError> {
        self.tensor_infos
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| InferenceError::TensorNotFound(name.to_string()))
    }

    /// Absolute byte offset where the tensor data section starts in the file.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Alignment value used for the data section.
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// All metadata keys.
    pub fn metadata_keys(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(|k| k.as_str())
    }

    /// Number of KV metadata entries.
    pub fn n_metadata(&self) -> usize {
        self.metadata.len()
    }

    // -- Tensor data access -------------------------------------------------

    /// Get the raw bytes for a tensor by index (zero-copy from mmap).
    ///
    /// The slice covers exactly the tensor's data (no padding).
    pub fn tensor_data(&self, index: usize) -> Result<&[u8], InferenceError> {
        let info = self.tensor_infos.get(index).ok_or_else(|| {
            InferenceError::TensorNotFound(format!("tensor index {} out of range", index))
        })?;
        let start = self.data_offset + info.offset;
        let size = info.byte_size();
        let end = start + size;

        if end as usize > self.mmap.len() {
            return Err(InferenceError::GgufParse(format!(
                "tensor '{}' data extends beyond file: offset={}, size={}, file_len={}",
                info.name,
                start,
                size,
                self.mmap.len()
            )));
        }

        Ok(&self.mmap[start as usize..end as usize])
    }

    /// Get the raw bytes for a tensor by name (zero-copy from mmap).
    pub fn tensor_data_by_name(&self, name: &str) -> Result<&[u8], InferenceError> {
        let (index, _) = self
            .find_tensor(name)
            .ok_or_else(|| InferenceError::TensorNotFound(name.to_string()))?;
        self.tensor_data(index)
    }

    /// Get a reference to the full memory-mapped file contents.
    pub fn mmap(&self) -> &memmap2::Mmap {
        &self.mmap
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("n_metadata", &self.metadata.len())
            .field("n_tensors", &self.tensor_infos.len())
            .field("data_offset", &self.data_offset)
            .field("alignment", &self.alignment)
            .field("file_size", &self.mmap.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Binary reader helpers
// ---------------------------------------------------------------------------

/// Read a little-endian u32.
fn read_u32<R: Read>(r: &mut R) -> Result<u32, InferenceError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64.
fn read_u64<R: Read>(r: &mut R) -> Result<u64, InferenceError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Read a little-endian i32.
fn read_i32<R: Read>(r: &mut R) -> Result<i32, InferenceError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

/// Read a little-endian i64.
fn read_i64<R: Read>(r: &mut R) -> Result<i64, InferenceError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a little-endian u16.
fn read_u16<R: Read>(r: &mut R) -> Result<u16, InferenceError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

/// Read a little-endian i16.
fn read_i16<R: Read>(r: &mut R) -> Result<i16, InferenceError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

/// Read a single byte as u8.
fn read_u8<R: Read>(r: &mut R) -> Result<u8, InferenceError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Read a single byte as i8.
fn read_i8<R: Read>(r: &mut R) -> Result<i8, InferenceError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
}

/// Read a little-endian f32.
fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, InferenceError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// Read a little-endian f64.
fn read_f64_le<R: Read>(r: &mut R) -> Result<f64, InferenceError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

/// Read a GGUF string: u64 length prefix + raw UTF-8 bytes (no null terminator).
fn read_gguf_string<R: Read>(r: &mut R) -> Result<String, InferenceError> {
    let len = read_u64(r)? as usize;
    if len > 1_000_000 {
        return Err(InferenceError::GgufParse(format!(
            "string length {} is suspiciously large",
            len
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| InferenceError::GgufParse(format!("invalid UTF-8: {}", e)))
}

/// Read a GGUF bool: stored as a single i8 (0 = false, nonzero = true).
fn read_gguf_bool<R: Read>(r: &mut R) -> Result<bool, InferenceError> {
    let v = read_i8(r)?;
    Ok(v != 0)
}

/// Read a single GGUF value of the given type.
fn read_gguf_value<R: Read>(
    r: &mut R,
    vtype: GgufValueType,
) -> Result<GgufValue, InferenceError> {
    match vtype {
        GgufValueType::Uint8 => Ok(GgufValue::U8(read_u8(r)?)),
        GgufValueType::Int8 => Ok(GgufValue::I8(read_i8(r)?)),
        GgufValueType::Uint16 => Ok(GgufValue::U16(read_u16(r)?)),
        GgufValueType::Int16 => Ok(GgufValue::I16(read_i16(r)?)),
        GgufValueType::Uint32 => Ok(GgufValue::U32(read_u32(r)?)),
        GgufValueType::Int32 => Ok(GgufValue::I32(read_i32(r)?)),
        GgufValueType::Float32 => Ok(GgufValue::F32(read_f32_le(r)?)),
        GgufValueType::Bool => Ok(GgufValue::Bool(read_gguf_bool(r)?)),
        GgufValueType::String => Ok(GgufValue::String(read_gguf_string(r)?)),
        GgufValueType::Uint64 => Ok(GgufValue::U64(read_u64(r)?)),
        GgufValueType::Int64 => Ok(GgufValue::I64(read_i64(r)?)),
        GgufValueType::Float64 => Ok(GgufValue::F64(read_f64_le(r)?)),
        GgufValueType::Array => {
            let elem_type_id = read_u32(r)?;
            let elem_type = GgufValueType::from_u32(elem_type_id)?;
            let count = read_u64(r)? as usize;

            if count > 100_000_000 {
                return Err(InferenceError::GgufParse(format!(
                    "array with {} elements is suspiciously large",
                    count
                )));
            }

            let mut elements = Vec::with_capacity(count);
            for _ in 0..count {
                elements.push(read_gguf_value(r, elem_type)?);
            }
            Ok(GgufValue::Array(elements))
        }
    }
}

/// Round `offset` up to the next multiple of `alignment`.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -- Binary reader helpers --

    #[test]
    fn test_read_u32() {
        let data: Vec<u8> = vec![0x47, 0x47, 0x55, 0x46]; // GGUF magic LE
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u32(&mut cursor).unwrap(), GGUF_MAGIC);
    }

    #[test]
    fn test_read_u64() {
        let data = 42u64.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u64(&mut cursor).unwrap(), 42);
    }

    #[test]
    fn test_read_i32() {
        let data = (-1i32).to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_i32(&mut cursor).unwrap(), -1);
    }

    #[test]
    fn test_read_f32() {
        let data = 3.14f32.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_f32_le(&mut cursor).unwrap();
        assert!((val - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_read_gguf_string() {
        // length=5, "hello"
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");
        let mut cursor = Cursor::new(data);
        assert_eq!(read_gguf_string(&mut cursor).unwrap(), "hello");
    }

    #[test]
    fn test_read_gguf_string_empty() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        assert_eq!(read_gguf_string(&mut cursor).unwrap(), "");
    }

    #[test]
    fn test_read_gguf_bool() {
        let mut cursor_false = Cursor::new(vec![0u8]);
        assert!(!read_gguf_bool(&mut cursor_false).unwrap());
        let mut cursor_true = Cursor::new(vec![1u8]);
        assert!(read_gguf_bool(&mut cursor_true).unwrap());
        // Nonzero also true
        let mut cursor_nonzero = Cursor::new(vec![42u8]);
        assert!(read_gguf_bool(&mut cursor_nonzero).unwrap());
    }

    // -- Alignment --

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(64, 32), 64);
    }

    // -- GgufValueType --

    #[test]
    fn test_value_type_from_u32() {
        assert!(matches!(
            GgufValueType::from_u32(0).unwrap(),
            GgufValueType::Uint8
        ));
        assert!(matches!(
            GgufValueType::from_u32(8).unwrap(),
            GgufValueType::String
        ));
        assert!(matches!(
            GgufValueType::from_u32(9).unwrap(),
            GgufValueType::Array
        ));
        assert!(matches!(
            GgufValueType::from_u32(12).unwrap(),
            GgufValueType::Float64
        ));
        assert!(GgufValueType::from_u32(13).is_err());
        assert!(GgufValueType::from_u32(999).is_err());
    }

    // -- read_gguf_value --

    #[test]
    fn test_read_gguf_value_u32() {
        let data = 42u32.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Uint32).unwrap();
        assert!(matches!(val, GgufValue::U32(42)));
    }

    #[test]
    fn test_read_gguf_value_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(b"foo");
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::String).unwrap();
        match val {
            GgufValue::String(s) => assert_eq!(s, "foo"),
            _ => panic!("expected String"),
        }
    }

    #[test]
    fn test_read_gguf_value_bool_true() {
        let data = vec![1u8];
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Bool).unwrap();
        assert!(matches!(val, GgufValue::Bool(true)));
    }

    #[test]
    fn test_read_gguf_value_array_of_u32() {
        let mut data = Vec::new();
        // Element type: U32 = 4
        data.extend_from_slice(&4u32.to_le_bytes());
        // Count: 3
        data.extend_from_slice(&3u64.to_le_bytes());
        // Elements: 10, 20, 30
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(&30u32.to_le_bytes());
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Array).unwrap();
        match val {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert!(matches!(arr[0], GgufValue::U32(10)));
                assert!(matches!(arr[1], GgufValue::U32(20)));
                assert!(matches!(arr[2], GgufValue::U32(30)));
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_read_gguf_value_array_of_strings() {
        let mut data = Vec::new();
        // Element type: String = 8
        data.extend_from_slice(&8u32.to_le_bytes());
        // Count: 2
        data.extend_from_slice(&2u64.to_le_bytes());
        // String "ab"
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"ab");
        // String "cd"
        data.extend_from_slice(&2u64.to_le_bytes());
        data.extend_from_slice(b"cd");
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Array).unwrap();
        match val {
            GgufValue::Array(arr) => {
                assert_eq!(arr.len(), 2);
                match &arr[0] {
                    GgufValue::String(s) => assert_eq!(s, "ab"),
                    _ => panic!("expected String"),
                }
                match &arr[1] {
                    GgufValue::String(s) => assert_eq!(s, "cd"),
                    _ => panic!("expected String"),
                }
            }
            _ => panic!("expected Array"),
        }
    }

    // -- Full GGUF file parsing (synthetic) --

    /// Build a minimal valid GGUF v3 file in memory with the given KV pairs
    /// and tensor info entries.
    fn build_gguf_file(
        kv_pairs: &[(&str, GgufValueType, &[u8])],
        tensors: &[(&str, &[u64], u32, u64, &[u8])], // (name, dims, dtype, offset, data)
        alignment: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version
        buf.extend_from_slice(&3u32.to_le_bytes());
        // n_tensors
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // n_kv
        buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());

        // KV pairs
        for &(key, _vtype, value_bytes) in kv_pairs {
            // Key string
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value (type + data, already serialized in value_bytes)
            buf.extend_from_slice(value_bytes);
        }

        // Tensor info
        for &(name, dims, dtype, offset, _data) in tensors {
            // Name string
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            // n_dims
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            // dims
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            // dtype
            buf.extend_from_slice(&dtype.to_le_bytes());
            // offset
            buf.extend_from_slice(&offset.to_le_bytes());
        }

        // Pad to alignment
        let pos = buf.len();
        let aligned = align_offset(pos as u64, alignment as u64) as usize;
        buf.resize(aligned, 0);

        // Tensor data
        for &(_name, _dims, _dtype, _offset, data) in tensors {
            buf.extend_from_slice(data);
            // Pad each tensor to alignment
            let pos = buf.len();
            let aligned = align_offset(pos as u64, alignment as u64) as usize;
            buf.resize(aligned, 0);
        }

        buf
    }

    /// Helper to serialize a U32 KV value (type_id + value).
    fn kv_u32(val: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF_TYPE_UINT32
        buf.extend_from_slice(&val.to_le_bytes());
        buf
    }

    /// Helper to serialize a String KV value (type_id + string).
    fn kv_string(val: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&8u32.to_le_bytes()); // GGUF_TYPE_STRING
        buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
        buf.extend_from_slice(val.as_bytes());
        buf
    }

    /// Helper to serialize an F32 KV value.
    fn kv_f32(val: f32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&6u32.to_le_bytes()); // GGUF_TYPE_FLOAT32
        buf.extend_from_slice(&val.to_le_bytes());
        buf
    }

    /// Helper to serialize a Bool KV value.
    fn kv_bool(val: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&7u32.to_le_bytes()); // GGUF_TYPE_BOOL
        buf.push(if val { 1 } else { 0 });
        buf
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let arch_kv = kv_string("gemma3");

        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &arch_kv)],
            &[],
            32,
        );

        // Write to temp file
        let dir = std::env::temp_dir();
        let path = dir.join("test_minimal.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.get_str("general.architecture"), Some("gemma3"));
        assert_eq!(gguf.n_tensors(), 0);
        assert_eq!(gguf.n_metadata(), 1);
        assert_eq!(gguf.alignment(), 32);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parse_gguf_with_kv_types() {
        let kv_arch = kv_string("llama");
        let kv_layers = kv_u32(24);
        let kv_eps = kv_f32(1e-6);
        let kv_causal = kv_bool(true);

        let file_bytes = build_gguf_file(
            &[
                ("general.architecture", GgufValueType::String, &kv_arch),
                ("llama.block_count", GgufValueType::Uint32, &kv_layers),
                (
                    "llama.attention.layer_norm_rms_epsilon",
                    GgufValueType::Float32,
                    &kv_eps,
                ),
                (
                    "llama.attention.causal",
                    GgufValueType::Bool,
                    &kv_causal,
                ),
            ],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_kv_types.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.get_str("general.architecture"), Some("llama"));
        assert_eq!(gguf.get_u32("llama.block_count"), Some(24));
        let eps = gguf.get_f32("llama.attention.layer_norm_rms_epsilon").unwrap();
        assert!((eps - 1e-6).abs() < 1e-10);
        assert_eq!(gguf.get_bool("llama.attention.causal"), Some(true));
        assert_eq!(gguf.n_metadata(), 4);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parse_gguf_with_tensor() {
        let kv_arch = kv_string("test");

        // Create a tensor with 4 F32 values (16 bytes)
        let tensor_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[(
                "test_tensor",
                &[4],                             // shape: [4]
                0,                                 // F32
                0,                                 // offset 0 relative to data section
                &tensor_data,
            )],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_tensor.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.n_tensors(), 1);

        let info = &gguf.tensor_infos()[0];
        assert_eq!(info.name, "test_tensor");
        assert_eq!(info.n_dims, 1);
        assert_eq!(info.dims, vec![4]);
        assert_eq!(info.dtype, GgufTensorType::F32);
        assert_eq!(info.offset, 0);
        assert_eq!(info.n_elements(), 4);
        assert_eq!(info.byte_size(), 16);

        // Read tensor data
        let data = gguf.tensor_data(0).unwrap();
        assert_eq!(data.len(), 16);
        let f0 = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(f0, 1.0);
        let f1 = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(f1, 2.0);

        // Read by name
        let data2 = gguf.tensor_data_by_name("test_tensor").unwrap();
        assert_eq!(data, data2);

        // Non-existent tensor
        assert!(gguf.tensor_data_by_name("nonexistent").is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_parse_gguf_2d_tensor() {
        let kv_arch = kv_string("test");

        // 2x3 F32 tensor = 6 elements = 24 bytes
        let tensor_data: Vec<u8> = (1..=6)
            .map(|i| i as f32)
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[(
                "weight",
                &[3, 2],  // shape: [3, 2] (GGUF uses column-major dims)
                0,        // F32
                0,
                &tensor_data,
            )],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_2d_tensor.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let info = &gguf.tensor_infos()[0];
        assert_eq!(info.name, "weight");
        assert_eq!(info.n_dims, 2);
        assert_eq!(info.dims, vec![3, 2]);
        assert_eq!(info.n_elements(), 6);
        assert_eq!(info.byte_size(), 24);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_invalid_magic() {
        let mut file_bytes = vec![0u8; 64];
        // Wrong magic
        file_bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_magic.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::InvalidMagic(m) => assert_eq!(m, 0xDEADBEEF),
            e => panic!("expected InvalidMagic, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unsupported_version() {
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        file_bytes.extend_from_slice(&99u32.to_le_bytes()); // version 99
        // Pad to make it a valid-enough file
        file_bytes.resize(64, 0);

        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_version.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::UnsupportedVersion(v) => assert_eq!(v, 99),
            e => panic!("expected UnsupportedVersion, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_require_str() {
        let kv_arch = kv_string("bert");
        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_require_str.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.require_str("general.architecture").unwrap(), "bert");
        assert!(gguf.require_str("nonexistent").is_err());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_find_tensor() {
        let kv_arch = kv_string("test");
        let tensor_data = vec![0u8; 16];

        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[("my_tensor", &[4], 0, 0, &tensor_data)],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_find_tensor.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let (idx, info) = gguf.find_tensor("my_tensor").unwrap();
        assert_eq!(idx, 0);
        assert_eq!(info.name, "my_tensor");

        assert!(gguf.find_tensor("nonexistent").is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_data_offset_alignment() {
        // With a very small header, data_offset should be aligned to 32
        let kv_arch = kv_string("t");
        let file_bytes = build_gguf_file(
            &[("a", GgufValueType::String, &kv_arch)],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_data_offset.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.data_offset() % 32, 0);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_gguf_value_type_name() {
        let v = GgufValue::U32(42);
        assert_eq!(v.type_name(), "U32");
        let v = GgufValue::String("hello".to_string());
        assert_eq!(v.type_name(), "String");
        let v = GgufValue::Array(vec![]);
        assert_eq!(v.type_name(), "Array");
    }

    #[test]
    fn test_parse_gguf_with_q8_0_tensor() {
        let kv_arch = kv_string("test");

        // Q8_0 tensor with 32 elements = 1 block = 34 bytes
        let tensor_data = vec![0u8; 34];

        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[(
                "quantized",
                &[32],  // 32 elements
                8,      // Q8_0
                0,
                &tensor_data,
            )],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_q8_0_tensor.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let info = &gguf.tensor_infos()[0];
        assert_eq!(info.dtype, GgufTensorType::Q8_0);
        assert_eq!(info.n_elements(), 32);
        assert_eq!(info.byte_size(), 34);

        let data = gguf.tensor_data(0).unwrap();
        assert_eq!(data.len(), 34);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_debug_impl() {
        let kv_arch = kv_string("test");
        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_debug.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let debug_str = format!("{:?}", gguf);
        assert!(debug_str.contains("GgufFile"));
        assert!(debug_str.contains("n_metadata"));

        std::fs::remove_file(&path).ok();
    }

    // ====================================================================
    // NEW TESTS: GGUF parser edge cases and coverage gaps
    // ====================================================================

    // -- Binary reader edge cases --

    #[test]
    fn test_read_u16() {
        // Exercises the u16 reader that was untested.
        let data = 0xABCDu16.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u16(&mut cursor).unwrap(), 0xABCD);
    }

    #[test]
    fn test_read_i16() {
        // Exercises the i16 reader that was untested.
        let data = (-1234i16).to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_i16(&mut cursor).unwrap(), -1234);
    }

    #[test]
    fn test_read_u8() {
        let data = vec![0xFFu8];
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u8(&mut cursor).unwrap(), 255);
    }

    #[test]
    fn test_read_i8() {
        // -128 as i8 is 0x80
        let data = vec![0x80u8];
        let mut cursor = Cursor::new(data);
        assert_eq!(read_i8(&mut cursor).unwrap(), -128);
    }

    #[test]
    fn test_read_i64() {
        let data = (-99i64).to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_i64(&mut cursor).unwrap(), -99);
    }

    #[test]
    fn test_read_f64() {
        let data = 2.718281828459045f64.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_f64_le(&mut cursor).unwrap();
        assert!((val - 2.718281828459045).abs() < 1e-12);
    }

    #[test]
    fn test_read_u32_truncated() {
        // Only 3 bytes available when 4 are needed -- should return IO error.
        let data = vec![0u8; 3];
        let mut cursor = Cursor::new(data);
        assert!(read_u32(&mut cursor).is_err());
    }

    #[test]
    fn test_read_u64_truncated() {
        // Only 4 bytes available when 8 are needed.
        let data = vec![0u8; 4];
        let mut cursor = Cursor::new(data);
        assert!(read_u64(&mut cursor).is_err());
    }

    // -- read_gguf_string edge cases --

    #[test]
    fn test_read_gguf_string_suspiciously_large() {
        // A string declaring > 1M length should be rejected.
        let mut data = Vec::new();
        data.extend_from_slice(&2_000_000u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        let result = read_gguf_string(&mut cursor);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("suspiciously large"));
    }

    #[test]
    fn test_read_gguf_string_invalid_utf8() {
        // Valid length but invalid UTF-8 bytes.
        let mut data = Vec::new();
        data.extend_from_slice(&3u64.to_le_bytes());
        data.push(0xFF);
        data.push(0xFE);
        data.push(0xFD);
        let mut cursor = Cursor::new(data);
        let result = read_gguf_string(&mut cursor);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("invalid UTF-8"));
    }

    // -- All 13 GgufValue types exercised individually --

    #[test]
    fn test_read_gguf_value_all_scalar_types() {
        // U8
        let mut cursor = Cursor::new(vec![42u8]);
        let val = read_gguf_value(&mut cursor, GgufValueType::Uint8).unwrap();
        assert!(matches!(val, GgufValue::U8(42)));

        // I8
        let mut cursor = Cursor::new(vec![0xFEu8]); // -2 as i8
        let val = read_gguf_value(&mut cursor, GgufValueType::Int8).unwrap();
        assert!(matches!(val, GgufValue::I8(-2)));

        // U16
        let data = 1234u16.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Uint16).unwrap();
        assert!(matches!(val, GgufValue::U16(1234)));

        // I16
        let data = (-567i16).to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Int16).unwrap();
        assert!(matches!(val, GgufValue::I16(-567)));

        // I32
        let data = (-99i32).to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Int32).unwrap();
        assert!(matches!(val, GgufValue::I32(-99)));

        // U64
        let data = 999999u64.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Uint64).unwrap();
        assert!(matches!(val, GgufValue::U64(999999)));

        // I64
        let data = (-123456789i64).to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Int64).unwrap();
        assert!(matches!(val, GgufValue::I64(-123456789)));

        // F32
        let data = 3.14f32.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Float32).unwrap();
        match val {
            GgufValue::F32(v) => assert!((v - 3.14).abs() < 1e-6),
            _ => panic!("expected F32"),
        }

        // F64
        let data = 2.71828f64.to_le_bytes().to_vec();
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Float64).unwrap();
        match val {
            GgufValue::F64(v) => assert!((v - 2.71828).abs() < 1e-10),
            _ => panic!("expected F64"),
        }

        // Bool false
        let mut cursor = Cursor::new(vec![0u8]);
        let val = read_gguf_value(&mut cursor, GgufValueType::Bool).unwrap();
        assert!(matches!(val, GgufValue::Bool(false)));
    }

    #[test]
    fn test_read_gguf_value_empty_array() {
        // Array with zero elements should parse successfully.
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes()); // element type: U32
        data.extend_from_slice(&0u64.to_le_bytes()); // count: 0
        let mut cursor = Cursor::new(data);
        let val = read_gguf_value(&mut cursor, GgufValueType::Array).unwrap();
        match val {
            GgufValue::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("expected empty Array"),
        }
    }

    #[test]
    fn test_read_gguf_value_array_suspiciously_large() {
        // An array claiming > 100M elements should be rejected.
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes()); // element type: U32
        data.extend_from_slice(&200_000_000u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        let result = read_gguf_value(&mut cursor, GgufValueType::Array);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("suspiciously large"));
    }

    #[test]
    fn test_read_gguf_value_array_unknown_element_type() {
        // Array with invalid element type ID should error.
        let mut data = Vec::new();
        data.extend_from_slice(&99u32.to_le_bytes()); // unknown element type
        data.extend_from_slice(&1u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        let result = read_gguf_value(&mut cursor, GgufValueType::Array);
        assert!(result.is_err());
    }

    // -- GgufValue::type_name completeness --

    #[test]
    fn test_gguf_value_type_name_all_variants() {
        // Ensures every variant has a type_name string.
        assert_eq!(GgufValue::U8(0).type_name(), "U8");
        assert_eq!(GgufValue::I8(0).type_name(), "I8");
        assert_eq!(GgufValue::U16(0).type_name(), "U16");
        assert_eq!(GgufValue::I16(0).type_name(), "I16");
        assert_eq!(GgufValue::U32(0).type_name(), "U32");
        assert_eq!(GgufValue::I32(0).type_name(), "I32");
        assert_eq!(GgufValue::U64(0).type_name(), "U64");
        assert_eq!(GgufValue::I64(0).type_name(), "I64");
        assert_eq!(GgufValue::F32(0.0).type_name(), "F32");
        assert_eq!(GgufValue::F64(0.0).type_name(), "F64");
        assert_eq!(GgufValue::Bool(false).type_name(), "Bool");
        assert_eq!(GgufValue::String("".to_string()).type_name(), "String");
        assert_eq!(GgufValue::Array(vec![]).type_name(), "Array");
    }

    // -- GgufValueType from_u32 complete coverage --

    #[test]
    fn test_value_type_from_u32_all_types() {
        // Ensure every valid type ID maps correctly.
        let expected = [
            (0, GgufValueType::Uint8),
            (1, GgufValueType::Int8),
            (2, GgufValueType::Uint16),
            (3, GgufValueType::Int16),
            (4, GgufValueType::Uint32),
            (5, GgufValueType::Int32),
            (6, GgufValueType::Float32),
            (7, GgufValueType::Bool),
            (8, GgufValueType::String),
            (9, GgufValueType::Array),
            (10, GgufValueType::Uint64),
            (11, GgufValueType::Int64),
            (12, GgufValueType::Float64),
        ];
        for (id, expected_type) in expected {
            assert_eq!(
                GgufValueType::from_u32(id).unwrap(),
                expected_type,
                "type id {} did not match",
                id
            );
        }
    }

    // -- Alignment edge cases --

    #[test]
    fn test_align_offset_power_of_two_alignments() {
        // Test with alignment values that are powers of 2.
        assert_eq!(align_offset(0, 1), 0);
        assert_eq!(align_offset(1, 1), 1);
        assert_eq!(align_offset(5, 1), 5);
        assert_eq!(align_offset(0, 64), 0);
        assert_eq!(align_offset(1, 64), 64);
        assert_eq!(align_offset(63, 64), 64);
        assert_eq!(align_offset(64, 64), 64);
        assert_eq!(align_offset(65, 64), 128);
        // 128-byte alignment
        assert_eq!(align_offset(100, 128), 128);
        assert_eq!(align_offset(128, 128), 128);
    }

    // -- Version 1 rejected --

    #[test]
    fn test_version_1_rejected() {
        // Version 1 of GGUF should be explicitly rejected.
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        file_bytes.extend_from_slice(&1u32.to_le_bytes()); // version 1
        file_bytes.resize(64, 0);

        let dir = std::env::temp_dir();
        let path = dir.join("test_version_1.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::UnsupportedVersion(1) => {}
            e => panic!("expected UnsupportedVersion(1), got {:?}", e),
        }
        std::fs::remove_file(&path).ok();
    }

    // -- Version 0 rejected --

    #[test]
    fn test_version_0_rejected() {
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        file_bytes.extend_from_slice(&0u32.to_le_bytes()); // version 0
        file_bytes.resize(64, 0);

        let dir = std::env::temp_dir();
        let path = dir.join("test_version_0.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::UnsupportedVersion(0) => {}
            e => panic!("expected UnsupportedVersion(0), got {:?}", e),
        }
        std::fs::remove_file(&path).ok();
    }

    // -- Version 2 accepted --

    #[test]
    fn test_version_2_accepted() {
        // Version 2 should be accepted (only version 0 and 1 are rejected).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // version 2
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_kv = 0
        // Pad to alignment
        let aligned = align_offset(buf.len() as u64, 32) as usize;
        buf.resize(aligned, 0);

        let dir = std::env::temp_dir();
        let path = dir.join("test_version_2.gguf");
        std::fs::write(&path, &buf).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_ok(), "Version 2 should be accepted");
        std::fs::remove_file(&path).ok();
    }

    // -- Non-existent file --

    #[test]
    fn test_open_nonexistent_file() {
        let result = GgufFile::open("/tmp/this_file_definitely_does_not_exist_12345.gguf");
        assert!(result.is_err());
        // Should be an IO error
        match result.unwrap_err() {
            InferenceError::Io(_) => {}
            e => panic!("expected Io error, got {:?}", e),
        }
    }

    // -- Truncated file (valid magic, but not enough data for header) --

    #[test]
    fn test_truncated_file_after_magic() {
        // File has valid magic but nothing else.
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());

        let dir = std::env::temp_dir();
        let path = dir.join("test_truncated_after_magic.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        std::fs::remove_file(&path).ok();
    }

    // -- Metadata getters with wrong types --

    #[test]
    fn test_getter_type_mismatch() {
        // A key stored as U32 should not be returned by get_str.
        let kv_arch = kv_string("test");
        let kv_layers = kv_u32(42);
        let file_bytes = build_gguf_file(
            &[
                ("general.architecture", GgufValueType::String, &kv_arch),
                ("layers", GgufValueType::Uint32, &kv_layers),
            ],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_getter_mismatch.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        // get_str on a U32 key returns None
        assert!(gguf.get_str("layers").is_none());
        // get_u32 on a String key returns None
        assert!(gguf.get_u32("general.architecture").is_none());
        // get_f32 on a U32 key returns None
        assert!(gguf.get_f32("layers").is_none());
        // get_bool on a U32 key returns None
        assert!(gguf.get_bool("layers").is_none());
        // get_i32 on a U32 key returns None
        assert!(gguf.get_i32("layers").is_none());
        // get_u64 on a U32 key returns None
        assert!(gguf.get_u64("layers").is_none());
        // Nonexistent key returns None for all getters
        assert!(gguf.get("missing").is_none());
        assert!(gguf.get_str("missing").is_none());
        assert!(gguf.get_u32("missing").is_none());

        std::fs::remove_file(&path).ok();
    }

    // -- require_u32 and require_f32 --

    #[test]
    fn test_require_u32_and_f32() {
        let kv_arch = kv_string("test");
        let kv_layers = kv_u32(16);
        let kv_eps = kv_f32(1e-5);
        let file_bytes = build_gguf_file(
            &[
                ("arch", GgufValueType::String, &kv_arch),
                ("layers", GgufValueType::Uint32, &kv_layers),
                ("eps", GgufValueType::Float32, &kv_eps),
            ],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_require_u32_f32.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.require_u32("layers").unwrap(), 16);
        assert!((gguf.require_f32("eps").unwrap() - 1e-5).abs() < 1e-10);
        // Missing key should error
        assert!(gguf.require_u32("nonexistent").is_err());
        assert!(gguf.require_f32("nonexistent").is_err());

        std::fs::remove_file(&path).ok();
    }

    // -- require_tensor --

    #[test]
    fn test_require_tensor() {
        let kv_arch = kv_string("test");
        let tensor_data = vec![0u8; 16]; // 4 F32 values

        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[("embed.weight", &[4], 0, 0, &tensor_data)],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_require_tensor.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let info = gguf.require_tensor("embed.weight").unwrap();
        assert_eq!(info.name, "embed.weight");
        // Missing tensor should error
        let result = gguf.require_tensor("missing.weight");
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::TensorNotFound(name) => {
                assert_eq!(name, "missing.weight");
            }
            e => panic!("expected TensorNotFound, got {:?}", e),
        }

        std::fs::remove_file(&path).ok();
    }

    // -- metadata_keys --

    #[test]
    fn test_metadata_keys() {
        let kv_arch = kv_string("test");
        let kv_layers = kv_u32(8);
        let file_bytes = build_gguf_file(
            &[
                ("general.architecture", GgufValueType::String, &kv_arch),
                ("block_count", GgufValueType::Uint32, &kv_layers),
            ],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_metadata_keys.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let mut keys: Vec<&str> = gguf.metadata_keys().collect();
        keys.sort();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"general.architecture"));
        assert!(keys.contains(&"block_count"));

        std::fs::remove_file(&path).ok();
    }

    // -- tensor_data out of bounds --

    #[test]
    fn test_tensor_data_index_out_of_range() {
        let kv_arch = kv_string("test");
        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_tensor_oob.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        // No tensors -- index 0 should be out of range
        let result = gguf.tensor_data(0);
        assert!(result.is_err());

        std::fs::remove_file(&path).ok();
    }

    // -- Multiple tensors --

    #[test]
    fn test_multiple_tensors() {
        let kv_arch = kv_string("test");

        // Two F32 tensors: first is 4 elements (16 bytes), second is 2 elements (8 bytes)
        let data1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let data2: Vec<u8> = [10.0f32, 20.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // Build file manually to handle two tensors with correct offsets.
        // The first tensor is at offset 0, the second at offset 32 (aligned from 16).
        let file_bytes = build_gguf_file(
            &[("general.architecture", GgufValueType::String, &kv_arch)],
            &[
                ("tensor_a", &[4], 0, 0, &data1),
                ("tensor_b", &[2], 0, 32, &data2),
            ],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_multiple_tensors.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        assert_eq!(gguf.n_tensors(), 2);

        let info0 = &gguf.tensor_infos()[0];
        assert_eq!(info0.name, "tensor_a");
        assert_eq!(info0.n_elements(), 4);

        let info1 = &gguf.tensor_infos()[1];
        assert_eq!(info1.name, "tensor_b");
        assert_eq!(info1.n_elements(), 2);

        // Read first tensor data
        let d0 = gguf.tensor_data(0).unwrap();
        assert_eq!(d0.len(), 16);
        let f0 = f32::from_le_bytes([d0[0], d0[1], d0[2], d0[3]]);
        assert_eq!(f0, 1.0);

        std::fs::remove_file(&path).ok();
    }

    // -- get_str_array and get_f32_array --

    #[test]
    fn test_get_str_array() {
        // Build a GGUF with a string array KV
        let mut kv_array = Vec::new();
        kv_array.extend_from_slice(&9u32.to_le_bytes()); // GGUF_TYPE_ARRAY
        kv_array.extend_from_slice(&8u32.to_le_bytes()); // element type: String
        kv_array.extend_from_slice(&2u64.to_le_bytes()); // count: 2
        // String "foo"
        kv_array.extend_from_slice(&3u64.to_le_bytes());
        kv_array.extend_from_slice(b"foo");
        // String "bar"
        kv_array.extend_from_slice(&3u64.to_le_bytes());
        kv_array.extend_from_slice(b"bar");

        let kv_arch = kv_string("test");
        let file_bytes = build_gguf_file(
            &[
                ("general.architecture", GgufValueType::String, &kv_arch),
                ("tokens", GgufValueType::Array, &kv_array),
            ],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_str_array.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let arr = gguf.get_str_array("tokens").unwrap();
        assert_eq!(arr, vec!["foo", "bar"]);
        // Non-existent key
        assert!(gguf.get_str_array("missing").is_none());
        // Wrong type key
        assert!(gguf.get_str_array("general.architecture").is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_get_f32_array() {
        // Build a GGUF with an f32 array KV
        let mut kv_array = Vec::new();
        kv_array.extend_from_slice(&9u32.to_le_bytes()); // GGUF_TYPE_ARRAY
        kv_array.extend_from_slice(&6u32.to_le_bytes()); // element type: F32
        kv_array.extend_from_slice(&3u64.to_le_bytes()); // count: 3
        kv_array.extend_from_slice(&1.5f32.to_le_bytes());
        kv_array.extend_from_slice(&2.5f32.to_le_bytes());
        kv_array.extend_from_slice(&3.5f32.to_le_bytes());

        let kv_arch = kv_string("test");
        let file_bytes = build_gguf_file(
            &[
                ("general.architecture", GgufValueType::String, &kv_arch),
                ("scores", GgufValueType::Array, &kv_array),
            ],
            &[],
            32,
        );

        let dir = std::env::temp_dir();
        let path = dir.join("test_f32_array.gguf");
        std::fs::write(&path, &file_bytes).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let arr = gguf.get_f32_array("scores").unwrap();
        assert_eq!(arr.len(), 3);
        assert!((arr[0] - 1.5).abs() < 1e-6);
        assert!((arr[1] - 2.5).abs() < 1e-6);
        assert!((arr[2] - 3.5).abs() < 1e-6);
        // Non-array key
        assert!(gguf.get_f32_array("general.architecture").is_none());

        std::fs::remove_file(&path).ok();
    }

    // -- Non-power-of-2 alignment is rejected --

    #[test]
    fn test_non_power_of_2_alignment_rejected() {
        // Build a GGUF file with general.alignment = 3 (not a power of 2)
        let kv_arch = kv_string("test");
        let kv_alignment = kv_u32(3); // NOT a power of 2

        // Manually build the file bytes (cannot use build_gguf_file because
        // it uses the alignment for padding; we need the KV to be invalid).
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_kv

        // KV 1: general.architecture = "test"
        let key1 = "general.architecture";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1.as_bytes());
        buf.extend_from_slice(&kv_arch);

        // KV 2: general.alignment = 3
        let key2 = "general.alignment";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2.as_bytes());
        buf.extend_from_slice(&kv_alignment);

        // Pad enough that the file is parseable
        buf.resize(256, 0);

        let dir = std::env::temp_dir();
        let path = dir.join("test_bad_alignment.gguf");
        std::fs::write(&path, &buf).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("power of 2"), "Error message: {}", err_msg);

        std::fs::remove_file(&path).ok();
    }

    // -- Zero alignment rejected --

    #[test]
    fn test_zero_alignment_rejected() {
        let kv_arch = kv_string("test");
        let kv_alignment = kv_u32(0); // alignment = 0

        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_kv

        let key1 = "general.architecture";
        buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
        buf.extend_from_slice(key1.as_bytes());
        buf.extend_from_slice(&kv_arch);

        let key2 = "general.alignment";
        buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
        buf.extend_from_slice(key2.as_bytes());
        buf.extend_from_slice(&kv_alignment);

        buf.resize(256, 0);

        let dir = std::env::temp_dir();
        let path = dir.join("test_zero_alignment.gguf");
        std::fs::write(&path, &buf).unwrap();

        let result = GgufFile::open(&path);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("power of 2"), "Error message: {}", err_msg);

        std::fs::remove_file(&path).ok();
    }

    // -- TensorInfo n_elements edge case --

    #[test]
    fn test_tensor_info_n_elements_empty_dims() {
        // A tensor with zero-length dims vec should have n_elements = 1
        // (product of empty range is 1, then max(1, 1) = 1)
        let info = TensorInfo {
            name: "scalar".to_string(),
            n_dims: 0,
            dims: vec![],
            dtype: GgufTensorType::F32,
            offset: 0,
        };
        assert_eq!(info.n_elements(), 1);
    }
}
