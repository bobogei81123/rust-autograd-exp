use ndarray::*;
use variable::*;
use tensor::*;
use std::rc::Rc;
use std::cell::RefCell;

pub trait Op {
    fn backward(&self, &ArrayD<f64>) -> Vec<ArrayD<f64>>;
    fn get_parents(&self) -> Vec<RefVar>;
}

mod add;
pub use self::add::*;

mod sum;
pub use self::sum::*;

mod constant;
pub use self::constant::*;
