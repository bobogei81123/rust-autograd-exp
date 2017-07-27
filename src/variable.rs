use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};

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

pub type RefVar = Rc<RefCell<Variable>>;

pub fn variable(array: ArrayD<f64>) -> RefVar {
    Rc::new(RefCell::new(Variable {
        tensor: Tensor::new(array),
        meta: VarMeta::new(),
        op: Box::new(op::ConstantOp {})
    }))
}

