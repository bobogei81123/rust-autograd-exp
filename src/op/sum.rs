use super::*;

struct SumOp {
    parent: RefVar,
}

impl Op for SumOp {
    fn get_parents(&self) -> Vec<RefVar> { vec![self.parent.clone()] }
    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let par = &self.parent;
        vec![ArrayD::<f64>::from_elem(par.data().shape(), 1.) * grad]
    }
}

impl SumOp {
    pub fn apply(v1: RefVar) -> RefVar {
        let parent = v1.clone();

        let var = Rc::new(Variable {
            tensor: Tensor::new(ArrayD::<f64>::zeros(vec![]) + arr0(v1.data().scalar_sum())),
            meta: VarMeta::new(),
            op: Box::new(SumOp {
                parent: parent,
            }),
        });
        println!("${} = sum(${})", var.id(), v1.id());
        var.into()
    }
}

pub fn sum(v1: RefVar) -> RefVar {
    SumOp::apply(v1)
}
