use ndarray::*;

pub enum Tensor {
    CpuTensor(ArrayD<f64>),
}

macro_rules! derive_tensor_binary_operator {
    ($trait:ident, $fun:ident, $op:tt) => {
        // x + y
        impl ::std::ops::$trait for Tensor {
            type Output = Tensor;
            
            #[allow(unreachable_patterns)]
            fn $fun(self, rhs: Tensor) -> Tensor {
                match (self, rhs) {
                    (Tensor::CpuTensor(x), Tensor::CpuTensor(y)) => { Tensor::CpuTensor(x $op y) }
                    _ => panic!("Tensor type mismatch")
                }
            }
        }
        // x + &y
        impl<'b> ::std::ops::$trait<&'b Tensor> for Tensor {
            type Output = Tensor;
            
            #[allow(unreachable_patterns)]
            fn $fun(self, rhs: &'b Tensor) -> Tensor {
                match (self, rhs) {
                    (Tensor::CpuTensor(ref x), &Tensor::CpuTensor(ref y)) => { Tensor::CpuTensor(x $op y) }
                    _ => panic!("Tensor type mismatch")
                }
            }
        }
        // &x + y
        impl<'a> ::std::ops::$trait<Tensor> for &'a Tensor {
            type Output = Tensor;
            
            #[allow(unreachable_patterns)]
            fn $fun(self, rhs: Tensor) -> Tensor {
                match (self, rhs) {
                    (&Tensor::CpuTensor(ref x), Tensor::CpuTensor(ref y)) => { Tensor::CpuTensor(x $op y) }
                    _ => panic!("Tensor type mismatch")
                }
            }
        }
        impl<'a, 'b> ::std::ops::$trait<&'b Tensor> for &'a Tensor {
            type Output = Tensor;
            
            #[allow(unreachable_patterns)]
            fn $fun(self, rhs: &'b Tensor) -> Tensor {
                match (self, rhs) {
                    (&Tensor::CpuTensor(ref x), &Tensor::CpuTensor(ref y)) => { Tensor::CpuTensor(x $op y) }
                    _ => panic!("Tensor type mismatch")
                }
            }
        }
    }
}

derive_tensor_binary_operator!(Add, add, +);
derive_tensor_binary_operator!(Sub, sub, -);
derive_tensor_binary_operator!(Mul, mul, *);
derive_tensor_binary_operator!(Div, div, /);

impl Tensor {
    pub fn new(array: ArrayD<f64>) -> Tensor {
        Tensor::CpuTensor(array)
    }
    pub fn data(&self) -> &ArrayD<f64> {
        match self {
            &Tensor::CpuTensor(ref t) => &t
        }
    }
}

//fn test() {
    //let x = Tensor::new(ArrayD::<f64>::zeros(vec![2, 3]));
//}
