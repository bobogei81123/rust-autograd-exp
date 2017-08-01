//#![feature(proc_macro)]
extern crate core;

#[macro_use]
extern crate ndarray;
pub use ndarray::*;

#[macro_use]
mod macro_utils;
pub use macro_utils::*;

mod variable;
pub use variable::*;

pub mod tensor;

mod op;
pub use op::*;

mod gradient;
pub use gradient::*;

#[cfg(test)]
mod tests {
    use super::*;
    extern crate itertools;
    use self::itertools::all;
    #[test]
    fn it_works() {
        let a = variable(dyarr![[1., 2.], [3., 4.]]);
        let b = &a * &a;
        let c = a + 1.;
        let d = c / b;
        let e = sum(d);

        assert!((e.data()[[]] - 3.5069444444444446).abs() < 1e-10);

        let res = &get_gradient(e)[&0];
        let correct_res = arr2(&[[-3. , -0.5], [-0.18518519, -0.09375]]);

        let err = correct_res - res;
        assert!(all(err.iter(), |x| x.abs() < 1e-6));
    }
}
