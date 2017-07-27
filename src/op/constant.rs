use super::*;

pub struct ConstantOp { }

impl Op for ConstantOp {
    #[allow(unused_variables)]
    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> { vec![] }
    fn get_parents(&self) -> Vec<RefVar> { vec![] }
}

