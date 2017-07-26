use ndarray::*;

pub enum Tensor {
    CpuTensor(CpuTensor),
}

pub struct CpuTensor {
    data: ArrayD<f64>,
}

impl Tensor {
    pub fn new(array: ArrayD<f64>) -> Tensor {
        Tensor::CpuTensor(CpuTensor { data: array })
    }
    pub fn get_data(&self) -> &ArrayD<f64> {
        match self {
            &Tensor::CpuTensor(ref t) => &t.data
        }
    }
}
