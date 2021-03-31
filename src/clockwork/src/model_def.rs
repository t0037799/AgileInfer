use crate::{
    c_api::TVMTensor,
    model::{Op, OpArg},
    pod_de::{self, PodType},
};
use libloading::Library;
use serde::Deserialize;

// Clockwork's POD serializer add a version number before every non-primitive type.
// For vector of non-primitive type, it adds the version number before the vector's size field.
// For simplicity, I just add a dummy field to simplify the design of POD deserializer.
#[derive(Deserialize, Debug)]
pub(crate) struct ModelDef {
    _dummy0: u32,
    weights_memory: usize,
    weights_memory_paged: usize,
    workspace_memory: usize,
    io_memory: usize,
    so_function: Vec<Vec<u8>>,
    _dummy1: u32,
    op_defs: Vec<OpDef>,
    _dummy2: u32,
    inputs: Vec<TensorDef>,
    _dummy3: u32,
    outputs: Vec<TensorDef>,
    configured_page_size: usize,
    _dummy4: u32,
    weights_pages: Vec<PageDef>,
}

#[derive(Deserialize, Debug)]
struct TensorDef {
    base_offset: usize,
    page: u32,
    page_offset: usize,
    size: usize,
    shape: Vec<isize>,
    code: u32,
    bits: u32,
    lanes: u32,
}

#[derive(Deserialize, Debug)]
struct OpDef {
    _dummy0: u32,
    inputs: Vec<TensorDef>,
    so_function: u32,
    _dummy1: u32,
    workspace: Vec<WorkspaceDef>,
}

#[derive(Deserialize, Debug)]
struct WorkspaceDef {
    page: u32,
    offset: usize,
    size: usize,
}

#[derive(Deserialize, Debug)]
struct PageDef {
    offset: usize,
    size: usize,
}

impl ModelDef {
    pub fn new(bytes: &[u8]) -> Self {
        pod_de::from_bytes(bytes, PodType::Clockwork).unwrap()
    }

    pub fn input_size(&self) -> usize {
        self.inputs.iter().fold(0, |acc, o| acc + o.size)
    }

    pub fn output_size(&self) -> usize {
        self.outputs.iter().fold(0, |acc, o| acc + o.size)
    }
    pub fn workspace_size(&self) -> usize {
        self.workspace_memory
    }
    pub fn weights_size(&self) -> usize {
        self.weights_memory
    }

    // map each op_def into runnable closures
    pub(crate) fn setup_ops(&self, so_module: &Library) -> Vec<Op> {
        self.op_defs
            .iter()
            .map(|op_def| {
                let ffi_name = &self.so_function[op_def.so_function as usize];
                let ffi_function: extern "C" fn(
                    *const *const TVMTensor, // &[&TVMTensor]
                    *const u32,              // &[TVMTypeCode]
                    i32,                     // the above slices' len
                ) -> i32 = *unsafe {
                    so_module
                        .get(ffi_name)
                        .expect("An FFI function to the library should exist.")
                };
                let op_args: Vec<_> = op_def
                    .inputs
                    .iter()
                    .map(|op_input| {
                        /*
                        let offset = op_input.page as usize * self.configured_page_size
                            + op_input.page_offset;
                            */
                        let shape = op_input.shape.clone();
                        OpArg::new(op_input.base_offset, shape)
                    })
                    .collect();
                Op::new(
                    ffi_function,
                    op_args,
                    op_def
                        .workspace
                        .iter()
                        .map(|def| {
                            (def.page as usize * self.configured_page_size + def.offset) as u64
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let buf = include_bytes!("../resources/resnet18_v2/model.1.clockwork");
        let model = ModelDef::new(buf);
        assert_eq!(model._dummy0, 1);
        assert_eq!(model._dummy1, 1);
        assert_eq!(model._dummy2, 1);
        assert_eq!(model._dummy3, 1);
        assert_eq!(model._dummy4, 1);
    }
}
