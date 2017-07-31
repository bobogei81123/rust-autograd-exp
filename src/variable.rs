use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use std::ops::*;
use std::convert::From;

use op;
use tensor::Tensor;
use ndarray::ArrayD;

static ID_COUNTER: AtomicUsize = ATOMIC_USIZE_INIT;

pub struct VarMeta {
    id: usize,
}

impl VarMeta {
    pub fn new() -> VarMeta {
        VarMeta {
            id: ID_COUNTER.fetch_add(1, Ordering::SeqCst),
        }
    }
}

pub struct Variable {
    pub tensor: Tensor,
    pub meta: VarMeta,
    pub op: Box<op::Op>,
}

impl Variable {
    //fn get_tensor(&self) -> &Tensor;
    pub fn data(&self) -> &ArrayD<f64> { &self.tensor.data() }
    //fn get_meta(&self) -> &VarMeta;
    pub fn id(&self) -> usize { self.meta.id }
}

//pub type RefVar = Rc<RefCell<Variable>>;
#[derive(Clone)]
pub struct RefVar(Rc<RefCell<Variable>>);

impl Deref for RefVar {
    type Target = Rc<RefCell<Variable>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Rc<RefCell<Variable>>> for RefVar {
    fn from(x: Rc<RefCell<Variable>>) -> RefVar {
        RefVar(x)
    }
}

pub fn variable(array: ArrayD<f64>) -> RefVar {
    Rc::new(RefCell::new(Variable {
        tensor: Tensor::new(array),
        meta: VarMeta::new(),
        op: Box::new(op::ConstantOp {})
    })).into()
}

macro_rules! impl_refvar_op {

    ($trait: ident, $fn: ident) => {
        impl $trait<RefVar> for RefVar {
            type Output = RefVar;

            fn $fn(self, other: RefVar) -> Self::Output {
                op::$fn(self, other)
            }
        }

        impl<'a> $trait<RefVar> for &'a RefVar {
            type Output = RefVar;

            fn $fn(self, other: RefVar) -> Self::Output {
                op::add(self.clone(), other)
            }
        }

        impl<'b> $trait<&'b RefVar> for RefVar {
            type Output = RefVar;

            fn $fn(self, other: &'b RefVar) -> Self::Output {
                op::$fn(self, other.clone())
            }
        }

        impl<'a, 'b> $trait<&'b RefVar> for &'a RefVar {
            type Output = RefVar;

            fn $fn(self, other: &'b RefVar) -> Self::Output {
                op::$fn(self.clone(), other.clone())
            }
        }
    }
}

impl_refvar_op!(Add, add);
impl_refvar_op!(Sub, sub);
impl_refvar_op!(Mul, mul);
impl_refvar_op!(Div, div);

impl Add<f64> for RefVar {
    type Output = RefVar;
    fn add(self, v: f64) -> RefVar {
        op::add_const(self, v)
    }
}
impl Add<RefVar> for f64 {
    type Output = RefVar;
    fn add(self, v: RefVar) -> RefVar {
        op::add_const(v, self)
    }
}
