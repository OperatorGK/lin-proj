#![feature(test)]

extern crate test;

#[allow(soft_unstable)]
#[cfg(test)]
mod bench {
    use crate::*;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    use super::test::{Bencher, black_box};
    use std::borrow::Borrow;

    #[bench]
    fn bench_francis(b: &mut Bencher) {
        let mut a = Array::random([100, 100], Uniform::new(-10., 10.));
        b.iter(|| {
            let q = schur_form_inplace(a.view_mut());
            black_box(q);
        }
        )
    }
}
