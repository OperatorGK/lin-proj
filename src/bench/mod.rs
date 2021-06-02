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
        let a = Array::random([50, 50], Uniform::new(-10., 10.));
        b.iter(|| {
            let (t, _) = schur_form(a.view()).unwrap();
            black_box(t);
        })
    }

    #[bench]
    fn bench_symmetric(b: &mut Bencher) {
        let mut a = Array::random([50, 50], Uniform::new(-10., 10.));
        for i in 0..50 {
            for j in 0..i {
                a[[i, j]] = a[[j, i]]
            }
        }

        let opts = QROptions {
            algorithm: QRAlgorithm::Symmetric,
            ..DEFAULT_OPTS
        };

        b.iter(|| {
            let (t, _) = schur_form_opts(a.view(), &opts).unwrap();
            black_box(t);
        })
    }
}
