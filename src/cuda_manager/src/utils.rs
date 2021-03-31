//! Basic utils

use cuda_driver_sys::cudaError_enum;

pub(crate) trait ToResult {
    type Error;
    fn to_result(&self) -> Result<(), Self::Error>;
}
impl ToResult for cudaError_enum {
    type Error = cudaError_enum;
    fn to_result(&self) -> Result<(), Self::Error> {
        match self {
            cudaError_enum::CUDA_SUCCESS => Ok(()),
            cudaError_enum::CUDA_ERROR_DEINITIALIZED => {
                log::warn!("CUDA already deinitialized, maybe the main thread is exited");
                Ok(())
            }
            &err => {
                log::warn!("Cuda Error {:?}", err);
                Err(err)
            }
        }
    }
}

/// Vec<T> to Vec<u8>
pub fn vec_to_bytes<T>(orig: Vec<T>) -> Vec<u8> {
    let slice = orig.leak();
    let size = std::mem::size_of::<T>() * slice.len();
    let ptr = slice.as_mut_ptr();
    unsafe { Vec::from_raw_parts(ptr as *mut u8, size, size) }
}

/// Vec<u8> to Vec<T>
pub fn bytes_to_vec<T>(orig: Vec<u8>) -> Vec<T> {
    let slice = orig.leak();
    let size = slice.len() / std::mem::size_of::<T>();
    let ptr = slice.as_mut_ptr();
    unsafe { Vec::from_raw_parts(ptr as *mut T, size, size) }
}
