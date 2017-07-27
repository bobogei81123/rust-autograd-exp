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
    #[test]
    fn it_works() {
        let a = variable(dyarr![[1., 2.], [3., 4.]]);
        let b = add(add(a.clone(), a.clone()), a.clone());
        let c = add(b.clone(), a.clone());
        let d = sum(c);
        //let b = ConstantVar::from_array(dyarr![[3., 4.], [5., 6.]]);
        //let c = add(a, b);
        //let d = sum(c);
        println!("Result: ${} = ${}", d.borrow().id(), d.borrow().data());

        let res = get_gradient(d);
        println!("Res = ${:?}", res);
        //assert!(false);
    }
}
