extern crate test;

#[allow(soft_unstable)]
#[cfg(test)]
mod bench {
    use crate::*;

    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use super::test::{black_box, Bencher};

    #[bench]
    fn bench_francis(b: &mut Bencher) {
        let a = Array::random([100, 100], Uniform::new(-10., 10.));
        b.iter(|| {
            let (t, _) = schur_form(a.view()).unwrap();
            black_box(t);
        })
    }
}
