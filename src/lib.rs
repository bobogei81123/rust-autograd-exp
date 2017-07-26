extern crate core;
extern crate ndarray;

#[macro_use]
mod macro_utils;

mod tensor;
use tensor::*;

use ndarray::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use std::collections::{HashMap, VecDeque};
use std::collections::hash_map;

type RefVar = Rc<RefCell<Variable>>;

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

pub trait Variable {
    fn get_tensor(&self) -> &Tensor;
    fn get_data(&self) -> &ArrayD<f64> { &self.get_tensor().get_data() }

    fn get_meta(&self) -> &VarMeta;
    fn get_id(&self) -> usize { self.get_meta().id }

    fn backward(&self, &ArrayD<f64>) -> Vec<ArrayD<f64>>;
    fn get_parents(&self) -> &Vec<RefVar>;
}

pub struct ConstantVar {
    tensor: Tensor,
    var_meta: VarMeta,
    parents: Vec<RefVar>,
}

impl Variable for ConstantVar {
    fn get_tensor(&self) -> &Tensor { &self.tensor }
    fn get_meta(&self) -> &VarMeta { &self.var_meta }

    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        vec![]
    }
    fn get_parents(&self) -> &Vec<RefVar> { &self.parents }
}

impl ConstantVar {
    fn array_to_refvar(array: ArrayD<f64>) -> RefVar {
        let var = Rc::new(RefCell::new(ConstantVar {
            tensor: Tensor::new(array),
            var_meta: VarMeta::new(),
            parents: vec![],
        }));
        {
            let vb = var.borrow();
            println!("${} = {}", vb.get_id(), vb.get_data());
        }
        var
    }
}

pub fn Variable(array: ArrayD<f64>) -> RefVar {
    ConstantVar::array_to_refvar(array)
}

struct AddVar {
    tensor: Tensor,
    var_meta: VarMeta,
    parents: Vec<RefVar>,
}

impl Variable for AddVar {
    fn get_tensor(&self) -> &Tensor { &self.tensor }
    fn get_meta(&self) -> &VarMeta { &self.var_meta }

    fn get_parents(&self) -> &Vec<RefVar> {
        &self.parents
    }
    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        self.get_parents().iter().map(|par| {
            ArrayD::<f64>::from_elem(par.borrow().get_data().shape(), 1.) * grad
        }).collect()
    }
}

fn add(v1: RefVar, v2: RefVar) -> RefVar {
    let _v1 = v1.borrow();
    let _v2 = v2.borrow();
    let parents = vec![v1.clone(), v2.clone()];

    let var = Rc::new(RefCell::new(AddVar {
        tensor: Tensor::new(_v1.get_data() + _v2.get_data()),
        var_meta: VarMeta::new(),
        parents: parents,
    }));
    println!("${} = ${} + ${}", 
             var.borrow().get_id(), _v1.get_id(), _v2.get_id());
    var
}

struct SumVar {
    tensor: Tensor,
    var_meta: VarMeta,
    parents: Vec<RefVar>,
}

impl Variable for SumVar {
    fn get_tensor(&self) -> &Tensor { &self.tensor }
    fn get_meta(&self) -> &VarMeta { &self.var_meta }

    fn get_parents(&self) -> &Vec<RefVar> { &self.parents }
    fn backward(&self, grad: &ArrayD<f64>) -> Vec<ArrayD<f64>> {
        let par = &self.get_parents()[0];
        vec![
             ArrayD::<f64>::from_elem(par.borrow().get_data().shape(), 1.) * grad
        ]
    }
}

pub fn sum(v1: RefVar) -> RefVar {
    let _v1 = v1.borrow();
    let parents = vec![v1.clone()];

    let var = Rc::new(RefCell::new(SumVar {
        tensor: Tensor::new(dyarr![_v1.get_data().scalar_sum()]),
        var_meta: VarMeta::new(),
        parents: parents,
    }));
    println!("${} = sum(${})", var.borrow().get_id(), _v1.get_id());
    var
}

pub fn get_gradient(target: RefVar) -> HashMap<usize, ArrayD<f64>> {
    let mut deps: HashMap<usize, usize> = HashMap::new();
    let mut queue: VecDeque<RefVar> = VecDeque::new();
    let mut grad_result: HashMap<usize, ArrayD<f64>> = HashMap::new();

    deps.insert(target.borrow().get_id(), 0);
    queue.push_back(target.clone());

    while !queue.is_empty() {
        let cur = queue.pop_front().unwrap();
        let cur = cur.borrow();

        grad_result.insert(cur.get_id(), ArrayD::zeros(cur.get_data().shape()));

        for par in cur.get_parents() {
            match deps.entry(par.borrow().get_id()) {
                hash_map::Entry::Occupied(mut x) => { *x.get_mut() += 1; }
                hash_map::Entry::Vacant(x) => {
                    x.insert(1);
                    queue.push_back(par.clone());
                }
            }
        }
    }

    let mut queue: VecDeque<RefVar> = VecDeque::new();

    grad_result.insert(target.borrow().get_id(), dyarr![1.]);
    queue.push_back(target.clone());

    while !queue.is_empty() {
        let cur = queue.pop_front().unwrap();
        let cur = cur.borrow();
        let my_grad = grad_result.get(&cur.get_id()).unwrap().clone();
        let back_list = cur.backward(&my_grad);

        for (par, grad) in cur.get_parents().iter().zip(back_list) {
            let pid = par.borrow().get_id();
            let mut prev_grad = grad_result.get_mut(&pid).unwrap();
            *prev_grad += &grad;

            let mut cnt = deps.get_mut(&pid).unwrap();
            *cnt -= 1;
            if *cnt == 0 {
                queue.push_back(par.clone());
            }
        }
    }

    grad_result
}

#[cfg(test)]
mod tests {
    use ndarray::*;
    use super::*;
    #[test]
    fn it_works() {
        let a = Variable(dyarr![[1., 2.], [3., 4.]]);
        let b = add(add(a.clone(), a.clone()), a.clone());
        let c = add(b.clone(), a.clone());
        let d = sum(c);
        //let b = ConstantVar::from_array(dyarr![[3., 4.], [5., 6.]]);
        //let c = add(a, b);
        //let d = sum(c);
        println!("Result: ${} = ${}", d.borrow().get_id(), d.borrow().get_data());

        let res = get_gradient(d);
        println!("Res = ${:?}", res);
        //assert!(false);
    }
}
