use super::*;

macro_rules! impl_basic_op_for_Op {
    ($op_name: ident, $fun_name: ident, $op: tt, $backward: expr) => {
        struct $op_name {
            parents: Vec<RefVar>,
        }

        impl Op for $op_name {
            fn get_parents(&self) -> Vec<RefVar> { self.parents.clone() }

            fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
                $backward(self, grad)
            }
        }

        impl $op_name {
            fn apply(v1: RefVar, v2: RefVar) -> RefVar {
                let parents = vec![v1.clone(), v2.clone()];

                let var = Rc::new(Variable {
                    tensor: Tensor::new(v1.data() $op v2.data()),
                    meta: VarMeta::new(),
                    op: Box::new($op_name {
                        parents: parents,
                    })
                });
                println!(concat!("${} = ${} ", stringify!($op), " ${}"), 
                         var.id(), v1.id(), v2.id());
                var.into()
            }
        }

        pub fn $fun_name(v1: RefVar, v2: RefVar) -> RefVar {
            $op_name::apply(v1, v2)
        }
    }
}

impl_basic_op_for_Op!(
    AddOp, add, +,
    |s: &AddOp, grad| s.parents.iter().map(|par| {
        ArrayD::<f64>::from_elem(par.data().shape(), 1.) * grad
    }).collect()
);

impl_basic_op_for_Op!(
    SubOp, sub, -,
    |s: &SubOp, grad| vec![
        ArrayD::<f64>::from_elem(s.parents[0].data().shape(), 1.) * grad,
        ArrayD::<f64>::from_elem(s.parents[1].data().shape(), -1.) * grad
    ]
);

impl_basic_op_for_Op!(
    MulOp, mul, *,
    |s: &MulOp, grad| vec![
        s.parents[1].data() * grad,
        s.parents[0].data() * grad
    ]
);

impl_basic_op_for_Op!(
    DivOp, div, /,
    |s: &DivOp, grad| vec![
        1.0 / s.parents[1].data(),
        s.parents[1].data().map(|x| -1.0 / (x * x))
        * s.parents[0].data() * grad
    ]
);

struct AddConstOp {
    parent: RefVar,
}

impl Op for AddConstOp {
    fn get_parents(&self) -> Vec<RefVar> {
        vec![self.parent.clone()]
    }

    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![grad.clone()]
    }
}

impl AddConstOp {
    fn apply(v1: RefVar, v2: f64) -> RefVar {
        Rc::new(Variable {
            tensor: Tensor::new(v1.data() + v2),
            meta: VarMeta::new(),
            op: Box::new(AddConstOp { parent: v1.clone() }), 
        }).into()
    }
}

pub fn add_const(v: RefVar, c: f64) -> RefVar {
    AddConstOp::apply(v, c)
}

/*
struct SubOp {
    parents: Vec<RefVar>,
}

impl Op for SubOp {
    fn get_parents(&self) -> Vec<RefVar> { self.parents.clone() }

    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![
            ArrayD::<f64>::from_elem(self.parents[0].borrow().data().shape(), 1.) * grad,
            ArrayD::<f64>::from_elem(self.parents[1].borrow().data().shape(), -1.) * grad
        ]
    }
}

impl SubOp {
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
        println!("${} = ${} - ${}", 
                 var.borrow().id(), v1.id(), v2.id());
        var.into()
    }
}

pub fn sup(v1: RefVar, v2: RefVar) -> RefVar {
    SubOp::apply(v1, v2)
}

struct MulOp {
    parents: Vec<RefVar>,
}

impl Op for MulOp {
    fn get_parents(&self) -> Vec<RefVar> { self.parents.clone() }

    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![
            self.parents[1].borrow().data() * grad,
            self.parents[0].borrow().data() * grad
        ]
    }
}

impl MulOp {
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
        println!("${} = ${} * ${}", 
                 var.borrow().id(), v1.id(), v2.id());
        var.into()
    }
}

pub fn mul(v1: RefVar, v2: RefVar) -> RefVar {
    MulOp::apply(v1, v2)
}

impl Op for DivOp {
    fn get_parents(&self) -> Vec<RefVar> { self.parents.clone() }

    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![
            self.parents[1].borrow().data() * grad,
            self.parents[0].borrow().data() * grad
        ]
    }
}

impl MulOp {
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
        println!("${} = ${} * ${}", 
                 var.borrow().id(), v1.id(), v2.id());
        var.into()
    }
}

pub fn mul(v1: RefVar, v2: RefVar) -> RefVar {
    MulOp::apply(v1, v2)
}

*/
