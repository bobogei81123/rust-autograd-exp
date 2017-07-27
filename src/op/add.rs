use super::*;

struct AddOp {
    parents: Vec<RefVar>,
}

impl Op for AddOp {
    fn get_parents(&self) -> Vec<RefVar> { self.parents.clone() }

    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        self.parents.iter().map(|par| {
            ArrayD::<f64>::from_elem(par.borrow().data().shape(), 1.) * grad
        }).collect()
    }
}

impl AddOp {
    fn apply(v1: RefVar, v2: RefVar) -> RefVar {
        let parents = vec![v1.clone(), v2.clone()];
        let v1 = v1.borrow();
        let v2 = v2.borrow();

        let var = Rc::new(RefCell::new(Variable {
            tensor: Tensor::new(v1.data() + v2.data()),
            meta: VarMeta::new(),
            op: Box::new(AddOp {
                parents: parents,
            })
        }));
        println!("${} = ${} + ${}", 
                 var.borrow().id(), v1.id(), v2.id());
        var
    }
}

pub fn add(v1: RefVar, v2: RefVar) -> RefVar {
    AddOp::apply(v1, v2)
}

