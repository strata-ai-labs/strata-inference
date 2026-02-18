//! Pre-allocated Metal buffer pool for zero-allocation decode.
//!
//! All intermediate buffers are allocated at engine init time. During decode,
//! `pool.get(slot)` is an O(1) array index returning the raw MTLBuffer pointer.

use crate::backend::metal::ffi::*;

/// Index into the flat buffer pool. Assigned at graph build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BufferSlot(pub u16);

/// Pre-allocated pool of reusable Metal buffers.
///
/// Created once at engine init with sizes determined by the graph builder.
/// Each slot holds one Metal buffer of a fixed byte size.
pub(crate) struct BufferPool {
    /// (raw MTLBuffer pointer, byte_size) per slot.
    buffers: Vec<(Id, usize)>,
}

// Metal shared-mode buffers can be accessed from any thread.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}

impl BufferPool {
    /// Allocate one Metal buffer per slot.
    ///
    /// # Safety
    /// `device` must be a valid MTLDevice pointer.
    pub(crate) unsafe fn new(device: Id, sels: &Selectors, slot_sizes: &[usize]) -> Self {
        let buffers = slot_sizes
            .iter()
            .map(|&size| {
                let buf = msg_send_new_buffer_length(
                    device,
                    sels.new_buffer_with_length,
                    size,
                    MTL_RESOURCE_STORAGE_MODE_SHARED,
                );
                assert!(!buf.is_null(), "Metal buffer allocation failed for size {}", size);
                (buf, size)
            })
            .collect();
        Self { buffers }
    }

    /// Get the raw MTLBuffer pointer for a slot. O(1) array index.
    #[inline]
    pub(crate) fn get(&self, slot: BufferSlot) -> Id {
        self.buffers[slot.0 as usize].0
    }

    /// Number of slots in the pool.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.buffers.len()
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        unsafe {
            let rel = sel_registerName(b"release\0".as_ptr() as _);
            for &(buf, _) in &self.buffers {
                if !buf.is_null() {
                    msg_send_void(buf, rel);
                }
            }
        }
    }
}
