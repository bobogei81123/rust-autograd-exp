use std::collections::{HashMap, VecDeque};
use std::collections::hash_map;
use super::*;

pub fn get_gradient(target: RefVar) -> HashMap<usize, ArrayD<f64>> {
    let mut deps: HashMap<usize, usize> = HashMap::new();
    let mut queue: VecDeque<RefVar> = VecDeque::new();
    let mut grad_result: HashMap<usize, ArrayD<f64>> = HashMap::new();

    deps.insert(target.id(), 0);
    queue.push_back(target.clone());

    while !queue.is_empty() {
        let cur = queue.pop_front().unwrap();
        let cur = cur;

        grad_result.insert(cur.id(), ArrayD::zeros(cur.data().shape()));

        for par in cur.op.get_parents() {
            match deps.entry(par.id()) {
                hash_map::Entry::Occupied(mut x) => { *x.get_mut() += 1; }
                hash_map::Entry::Vacant(x) => {
                    x.insert(1);
                    queue.push_back(par.clone());
                }
            }
        }
    }

    let mut queue: VecDeque<RefVar> = VecDeque::new();

    grad_result.insert(target.id(), ArrayD::<f64>::zeros(vec![]) + arr0(1.));
    queue.push_back(target.clone());

    while !queue.is_empty() {
        let cur = queue.pop_front().unwrap();
        let my_grad = grad_result.get(&cur.id()).unwrap().clone();
        let back_list = cur.op.backward(&my_grad);

        for (par, grad) in cur.op.get_parents().iter().zip(back_list) {
            let pid = par.id();
            let mut prev_grad = grad_result.get_mut(&pid).unwrap();
            *prev_grad += &grad;
            //println!("{}: {} += {}",  cur.id(), pid, &grad);

            let mut cnt = deps.get_mut(&pid).unwrap();
            *cnt -= 1;
            if *cnt == 0 {
                queue.push_back(par.clone());
            }
        }
    }

    grad_result
}

