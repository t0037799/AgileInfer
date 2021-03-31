use crate::model::{CudaFunction, CudaInfo, WORKSPACE};
use serde::Deserialize;
use std::ffi::CStr;
#[allow(dead_code)]
pub(crate) enum TVMDeviceType {
    CPU = 1,
    GPU = 2,
}

#[allow(dead_code)]
#[repr(u32)]
#[derive(Debug)]
pub(crate) enum TVMTypeCode {
    Int = 0,
    UInt = 1,
    Float = 2,
    Handle = 3,
    Tensor = 7,
}
#[repr(C)]
pub(crate) struct TVMTensor {
    pub ptr: u64,
    pub ctx: TVMContext,
    pub ndim: i32,
    pub dtype: TVMDataType,
    pub shape: *const isize,
    pub stride: *const isize,
    pub byte_offset: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct TVMContext {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug, Deserialize)]
pub(crate) struct TVMDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[allow(dead_code)]
pub(crate) extern "C" fn tvm_func_call(
    function: *mut CudaFunction,
    args: *mut u64,
    args_code: *const TVMTypeCode,
    num_args: usize,
) -> i32 {
    unsafe {
        let args = std::slice::from_raw_parts_mut(args, num_args);
        let args_code = std::slice::from_raw_parts(args_code, num_args);
        let function = Box::from_raw(function);
        function.as_ref().call(args, args_code);
        Box::leak(function);
    }
    0
}

#[allow(dead_code)]
pub(crate) extern "C" fn tvm_backend_alloc(
    _device_type: u64,
    _device_id: u64,
    _size: usize,
) -> u64 {
    WORKSPACE.with(|w| w.borrow_mut().allocate())
}

#[allow(dead_code)]
pub(crate) extern "C" fn tvm_backend_free(_device_type: u64, _device_id: u64, _ptr: u64) -> i32 {
    0
}

#[allow(dead_code)]
pub(crate) extern "C" fn tvm_backend_get_func_from_env(
    cuda_info: *const CudaInfo,
    function_name: *const i8,
    function: *mut *const CudaFunction,
) -> i32 {
    unsafe {
        let function_name = CStr::from_ptr(function_name).to_bytes().to_vec();
        let cuda_info = &*cuda_info;
        *function = Box::leak(cuda_info.get_function(&function_name));
        function_name.leak();
    }
    0
}

#[allow(dead_code)]
pub(crate) extern "C" fn tvm_set_error() {
    panic!("Error from TVM library")
}
