macro_rules! dyarr {
    ($($x: tt)*) => {
        {
            let a = array![$($x)*];
            let b = ArrayD::<f64>::zeros(IxDyn(a.shape()));
            b + a
        }
    }
}
