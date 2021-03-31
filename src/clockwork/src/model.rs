use crate::{
    c_api::{TVMContext, TVMDataType, TVMDeviceType, TVMTensor, TVMTypeCode},
    model_def::ModelDef,
    pod_de::{Deserializer, PodType},
};
use cuda_manager::{CudaArgument, CudaKernel, CudaOperation, Handle};
use libloading::Library;
use serde::Deserialize;
use std::{
    cell::RefCell, collections::HashMap, error::Error, ffi::OsStr, fs::File, io::prelude::*,
    path::Path,
};

thread_local! {
pub(crate) static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace {
    ws: vec![],
    workspace_pointer: 0,
    handle: None,
    queue_id: 0,
});
}

pub(crate) struct Workspace {
    ws: Vec<u64>,
    workspace_pointer: u64,
    handle: Option<Handle>,
    queue_id: usize,
}

impl Workspace {
    fn init(&mut self, workspace_pointer: u64, handle: Handle, queue_id: usize) {
        self.workspace_pointer = workspace_pointer;
        self.handle = Some(handle);
        self.queue_id = queue_id;
    }

    pub fn init_allocate(&mut self, ws: Vec<u64>) {
        self.ws = ws;
    }

    pub fn allocate(&mut self) -> u64 {
        self.ws.pop().unwrap() + self.workspace_pointer
    }
}

pub struct Model {
    #[allow(dead_code)]
    so_module: Library,
    ops: Vec<Op>,
    cuda_info: CudaInfo,
    model_def: ModelDef,
}

pub(crate) struct CudaInfo {
    thread_axis_mapping: HashMap<Vec<u8>, Vec<usize>>,
    function_id_map: HashMap<Vec<u8>, usize>,
}

pub(crate) struct CudaFunction {
    function_id: usize,
    thread_axis_mapping: Vec<usize>,
}

impl CudaFunction {
    pub fn call(&self, args: &mut [u64], args_code: &[TVMTypeCode]) {
        if self.function_id == usize::MAX {
            return;
        }
        WORKSPACE.with(|w| {
            let w = w.borrow();
            let queue_id = w.queue_id;
            let handle = w.handle.as_ref().unwrap().clone();
            let mut thread_axis = [1; 6];
            let mut cuda_args = vec![];
            let mut thread_axis_args = vec![];
            for i in 0..args.len() {
                match args_code[i] {
                    TVMTypeCode::Handle => {
                        cuda_args.push(CudaArgument::Pointer(args[i]));
                    }
                    TVMTypeCode::Int => {
                        thread_axis_args.push(args[i] as u32);
                    }
                    _ => {
                        panic!("?")
                    }
                }
            }
            for i in 0..thread_axis_args.len() {
                thread_axis[self.thread_axis_mapping[i]] = thread_axis_args[i];
            }
            handle
                .submit_cuda_op(
                    queue_id,
                    CudaOperation::CudaKernel(
                        CudaKernel::new(self.function_id, thread_axis),
                        cuda_args,
                    ),
                )
                .unwrap();
        });
    }
}

impl CudaInfo {
    pub fn get_function(&self, function_name: &[u8]) -> Box<CudaFunction> {
        let ss = std::str::from_utf8(function_name).unwrap();
        if ss == "__tvm_set_device" {
            return Box::new(CudaFunction {
                function_id: usize::MAX,
                thread_axis_mapping: vec![],
            });
        }
        let function_id = self.function_id_map[function_name];
        let thread_axis_mapping = self.thread_axis_mapping[function_name].clone();
        Box::new(CudaFunction {
            function_id,
            thread_axis_mapping,
        })
    }
}

impl Model {
    pub fn new<P: AsRef<Path>, Q: AsRef<OsStr>>(
        model_def_path: P,
        model_lib_path: Q,
        handle: Handle,
    ) -> Result<Model, Box<dyn Error>> {
        let so_module = Library::new(model_lib_path)?;
        let mut f = File::open(model_def_path)?;
        let mut buf = vec![];
        f.read_to_end(&mut buf)?;
        let model_def = ModelDef::new(&buf);
        let ops = model_def.setup_ops(&so_module);
        let mut model = Model {
            so_module,
            ops,
            model_def,
            cuda_info: CudaInfo {
                thread_axis_mapping: HashMap::new(),
                function_id_map: HashMap::new(),
            },
        };
        model.setup_cuda_module(handle);
        model.setup_ffi();
        Ok(model)
    }

    fn setup_ffi(&self) {
        unsafe {
            if let Ok(tvm_fptr) = self.so_module.get::<*mut *const ()>(b"__TVMFuncCall") {
                **tvm_fptr = super::c_api::tvm_func_call as *const ();
            }
            if let Ok(tvm_fptr) = self
                .so_module
                .get::<*mut *const ()>(b"__TVMBackendGetFuncFromEnv")
            {
                **tvm_fptr = super::c_api::tvm_backend_get_func_from_env as *const ();
            }
            if let Ok(tvm_fptr) = self
                .so_module
                .get::<*mut *const ()>(b"__TVMBackendAllocWorkspace")
            {
                **tvm_fptr = super::c_api::tvm_backend_alloc as *const ();
            }
            if let Ok(tvm_fptr) = self
                .so_module
                .get::<*mut *const ()>(b"__TVMBackendFreeWorkspace")
            {
                **tvm_fptr = super::c_api::tvm_backend_free as *const ();
            }
            if let Ok(tvm_fptr) = self
                .so_module
                .get::<*mut *const ()>(b"__TVMAPISetLastError")
            {
                **tvm_fptr = super::c_api::tvm_set_error as *const ();
            }
        }
    }
    fn setup_cuda_module(&mut self, handle: Handle) {
        let device_blob = unsafe {
            let tvm_dev_mblob = *self
                .so_module
                .get::<*const u8>(b"__tvm_dev_mblob")
                .expect("Should contain Cuda module in So Moudle");
            let blob_ptr = tvm_dev_mblob.wrapping_add(8);
            let size_ptr = tvm_dev_mblob as *const usize;
            std::slice::from_raw_parts(blob_ptr, *size_ptr)
        };
        let mut deserializer = Deserializer::from_bytes(device_blob, PodType::TVM);
        let size = usize::deserialize(&mut deserializer).unwrap();
        for _ in 0..size {
            let tkey = Vec::<u8>::deserialize(&mut deserializer).unwrap();
            let tkey = String::from_utf8(tkey).unwrap();
            match tkey.as_str() {
                "_lib" => {}
                "_import_tree" => {
                    let _row_ptr = Vec::<u64>::deserialize(&mut deserializer).unwrap();
                    let _child_indices = Vec::<u64>::deserialize(&mut deserializer).unwrap();
                }
                "cuda" => {
                    let _fmt = Vec::<u8>::deserialize(&mut deserializer).unwrap();
                    let fmap =
                        HashMap::<Vec<u8>, FunctionInfo>::deserialize(&mut deserializer).unwrap();
                    let mut module_data = Vec::<u8>::deserialize(&mut deserializer).unwrap();
                    // cuModuleLoadData accept NULL-terminated string
                    // push additional 0 to terminate in case the module contains no terminated
                    // NULL byte.
                    module_data.push(0);
                    let module_id = handle.load_module(module_data).unwrap();
                    let function_id_map = fmap
                        .iter()
                        .map(|(name, _)| {
                            let mut name_null = name.clone();
                            name_null.push(0);
                            let function_id = handle.get_function(module_id, &name_null).unwrap();
                            (name.clone(), function_id)
                        })
                        .collect();
                    // create a thread_axis_tags mapping for each kernel
                    let thread_axis_mapping: HashMap<Vec<u8>, Vec<usize>> = fmap
                        .iter()
                        .map(|(name, info)| {
                            let mapping: Vec<usize> = info
                                .thread_axis_tags
                                .iter()
                                .map(|tag| {
                                    let tag = std::str::from_utf8(tag).unwrap();
                                    let idx = if tag.starts_with("blockIdx.") {
                                        tag.as_bytes()[9] - b'x'
                                    } else if tag.starts_with("threadIdx.") {
                                        tag.as_bytes()[10] - b'x' + 3
                                    } else {
                                        unimplemented!("Unknown thread axis tags {}", tag)
                                    };
                                    idx as usize
                                })
                                .collect();
                            (name.clone(), mapping)
                        })
                        .collect();
                    self.cuda_info.function_id_map = function_id_map;
                    self.cuda_info.thread_axis_mapping = thread_axis_mapping;
                }
                _ => {
                    panic!("Unsupported module");
                }
            }
        }
    }
    pub fn weight_size(&self) -> usize {
        self.model_def.weights_size()
    }
    pub fn input_size(&self) -> usize {
        self.model_def.input_size()
    }
    pub fn output_size(&self) -> usize {
        self.model_def.output_size()
    }
    pub fn workspace_size(&self) -> usize {
        self.model_def.workspace_size()
    }
    pub fn weight_offset(&self) -> usize {
        0
    }
    pub fn input_offset(&self) -> usize {
        self.weight_offset() + self.weight_size()
    }
    pub fn output_offset(&self) -> usize {
        self.input_offset() + self.input_size()
    }
    pub fn workspace_offset(&self) -> usize {
        self.output_offset() + self.output_size()
    }

    pub fn inference(&self, handle: Handle, queue_id: usize, pointer: u64) {
        unsafe {
            if let Ok(tvm_module_ctx) = self.so_module.get::<*mut *const ()>(b"__tvm_module_ctx") {
                **tvm_module_ctx = &self.cuda_info as *const _ as *const _;
            }
            WORKSPACE.with(|w| {
                w.borrow_mut()
                    .init(pointer + self.workspace_offset() as u64, handle, queue_id);
                for op in &self.ops {
                    w.borrow_mut().init_allocate(op.workspace.clone());
                    op.call(pointer);
                }
            });
        }
    }
}

#[derive(Deserialize)]
struct FunctionInfo {
    _name: Vec<u8>,
    _arg_types: Vec<TVMDataType>,
    thread_axis_tags: Vec<Vec<u8>>,
}

pub(crate) struct OpArg {
    offset: usize,
    shape: Vec<isize>,
}

impl OpArg {
    pub(crate) fn new(offset: usize, shape: Vec<isize>) -> Self {
        OpArg { offset, shape }
    }
}
pub(crate) struct Op {
    ffi_function: unsafe extern "C" fn(*const *const TVMTensor, *const u32, i32) -> i32,
    op_args: Vec<OpArg>,
    workspace: Vec<u64>,
}

impl Op {
    pub(crate) fn new(
        ffi_function: unsafe extern "C" fn(*const *const TVMTensor, *const u32, i32) -> i32,
        op_args: Vec<OpArg>,
        workspace: Vec<u64>,
    ) -> Self {
        Op {
            ffi_function,
            op_args,
            workspace,
        }
    }
}

impl Op {
    fn call(&self, data: u64) {
        let tvm_tensors: Vec<_> = self
            .op_args
            .iter()
            .map(|op_arg| TVMTensor {
                ptr: data + op_arg.offset as u64,
                ctx: TVMContext {
                    device_type: TVMDeviceType::GPU as i32,
                    device_id: 0,
                },
                ndim: op_arg.shape.len() as i32,
                dtype: TVMDataType {
                    code: 2,
                    bits: 32,
                    lanes: 1,
                },
                shape: op_arg.shape.as_ptr() as *const isize,
                stride: std::ptr::null(),
                byte_offset: 0,
            })
            .collect();
        let args: Vec<_> = tvm_tensors.iter().map(|t| t as *const TVMTensor).collect();
        let type_codes = vec![TVMTypeCode::Tensor as u32; args.len()];
        let ffi_function = self.ffi_function;
        unsafe { ffi_function(args.as_ptr(), type_codes.as_ptr(), args.len() as i32) };
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crossbeam::channel;
    use cuda_manager::utils::*;
    use cuda_manager::*;
    use serial_test::serial;
    use std::sync::Arc;
    use std::time::Duration;

    #[serial]
    #[test]
    fn test_inference() {
        let params = include_bytes!("../resources/resnet18_v2/model.clockwork_params").to_vec();
        let params = Arc::new(params);
        let model_def = "resources/resnet18_v2/model.1.clockwork";
        let model_lib = "resources/resnet18_v2/model.1.so";
        let gpu_id = 0;
        let cuda_manager = CudaManager::from_gpu_ids(&[gpu_id]).unwrap();
        let handle = cuda_manager.run();
        let model = Model::new(model_def, model_lib, handle.clone()).unwrap();
        let input_size = model.input_size();
        let output_size = model.output_size();
        let weight_size = model.weight_size();
        let workspace_size = model.workspace_size();
        let total_size = input_size + output_size + weight_size + workspace_size;

        let pointer = handle.allocate(total_size).unwrap();
        handle
            .copy_to_all_devices(pointer, Arc::clone(&params))
            .unwrap();

        let queue_id = handle.create_op_queue(gpu_id, false).unwrap();
        let input_buffer = include_bytes!("../resources/inputs/n01768244_309.JPEG").to_vec();
        let output_buffer = vec_to_bytes(vec![0.0f32; 1000]);
        let input_id = 0;
        handle.insert_host_memory(queue_id, input_id, input_buffer);
        let output_id = 1;
        handle.insert_host_memory(queue_id, output_id, output_buffer);
        handle
            .submit_cuda_op(
                queue_id,
                CudaOperation::MemcpyHtoD(
                    pointer + model.input_offset() as u64,
                    input_id,
                    input_size,
                ),
            )
            .unwrap();
        model.inference(handle.clone(), queue_id, pointer);
        handle
            .submit_cuda_op(
                queue_id,
                CudaOperation::MemcpyDtoH(
                    pointer + model.output_offset() as u64,
                    output_id,
                    output_size,
                ),
            )
            .unwrap();
        let (tx, rx) = channel::unbounded();
        handle
            .start_op_queue(queue_id, tx, Duration::from_secs(0), true)
            .unwrap();
        let (result, mut host_memories) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert_eq!(result, QueueResult::Finish);
        let output = bytes_to_vec::<f32>(host_memories.remove(&output_id).unwrap());
        let mut output: Vec<_> = output.iter().zip(0..).collect();
        output.sort_by(|(l, _), (r, _)| r.partial_cmp(l).unwrap());
        assert_eq!(
            output[..5].iter().map(|(_, o)| *o).collect::<Vec<_>>(),
            vec![921, 838, 631, 551, 798]
        );
    }
}
// output[921] = 7.81953
// output[838] = 7.54152
// output[631] = 6.66503
// output[551] = 6.62357
// output[798] = 6.42082
