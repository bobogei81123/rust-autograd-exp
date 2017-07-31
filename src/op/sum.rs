use super::*;

struct SumOp {
    parent: RefVar,
}

impl Op for SumOp {
    fn get_parents(&self) -> Vec<RefVar> { vec![self.parent.clone()] }
    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let par = &self.parent;
        vec![ArrayD::<f64>::from_elem(par.borrow().data().shape(), 1.) * grad]
    }
}

impl SumOp {
    pub fn apply(v1: RefVar) -> RefVar {
        let parent = v1.clone();
        let v1 = v1.borrow();

        let var = Rc::new(RefCell::new(Variable {
            tensor: Tensor::new(ArrayD::<f64>::zeros(vec![]) + arr0(v1.data().scalar_sum())),
            meta: VarMeta::new(),
            op: Box::new(SumOp {
                parent: parent,
            }),
        }));
        println!("${} = sum(${})", var.borrow().id(), v1.id());
        var.into()
    }
}

pub fn sum(v1: RefVar) -> RefVar {
    SumOp::apply(v1)
}
