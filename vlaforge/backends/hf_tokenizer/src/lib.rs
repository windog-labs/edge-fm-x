//! Thin ownership/error boundary around Hugging Face Tokenizers, not a tokenizer implementation.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use tokenizers::Tokenizer;

#[repr(C)]
pub struct TokenIds {
    data: *mut u32,
    size: usize,
}

#[no_mangle]
pub unsafe extern "C" fn vlaforge_hf_tokenizer_new(
    data: *const u8,
    size: usize,
    output: *mut *mut c_void,
) -> u32 {
    if output.is_null() {
        return 1;
    }
    *output = ptr::null_mut();
    if data.is_null() || size == 0 || size > (128 << 20) {
        return 1;
    }
    match catch_unwind(AssertUnwindSafe(|| Tokenizer::from_bytes(slice::from_raw_parts(data, size)))) {
        Ok(Ok(tokenizer)) => {
            *output = Box::into_raw(Box::new(tokenizer)).cast();
            0
        }
        _ => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vlaforge_hf_tokenizer_encode(
    handle: *mut c_void,
    data: *const u8,
    size: usize,
    special: u8,
    output: *mut TokenIds,
) -> u32 {
    if output.is_null() {
        return 1;
    }
    *output = TokenIds { data: ptr::null_mut(), size: 0 };
    if handle.is_null() || (data.is_null() && size != 0) || size > (1 << 20) || special > 1 {
        return 1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let bytes = if size == 0 { &[] } else { slice::from_raw_parts(data, size) };
        let text = std::str::from_utf8(bytes).map_err(|_| ())?;
        let encoded = (&*handle.cast::<Tokenizer>()).encode(text, special != 0).map_err(|_| ())?;
        Ok::<_, ()>(encoded.get_ids().to_vec().into_boxed_slice())
    }));
    match result {
        Ok(Ok(ids)) => {
            (*output).size = ids.len();
            (*output).data = Box::into_raw(ids).cast();
            0
        }
        _ => 1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vlaforge_hf_tokenizer_free_ids(ids: TokenIds) {
    if !ids.data.is_null() {
        drop(Box::from_raw(ptr::slice_from_raw_parts_mut(ids.data, ids.size)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn vlaforge_hf_tokenizer_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle.cast::<Tokenizer>()));
    }
}
