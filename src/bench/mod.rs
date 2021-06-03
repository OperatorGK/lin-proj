extern crate test;

#[allow(soft_unstable)]
#[cfg(test)]
mod bench {
    use crate::*;

    use ndarray::Array;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use super::test::{black_box, Bencher};

    const BENCH_SIZE: usize = 155;

    #[bench]
    fn bench_rng(b: &mut Bencher) {
        b.iter(|| {
            let a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            black_box(&a);
        })
    }

    #[bench]
    fn bench_qr(b: &mut Bencher) {
        b.iter(|| {
            let a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            let (q, _) = qr_decomposition(a.view()).unwrap();
            black_box(&q);
        })
    }

    #[bench]
    fn bench_hess(b: &mut Bencher) {
        b.iter(|| {
            let a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            let (h, _) = hessenberg_form(a.view()).unwrap();
            black_box(&h);
        })
    }

    #[bench]
    fn bench_francis(b: &mut Bencher) {
        b.iter(|| {
            let a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            let (t, _) = schur_form(a.view()).unwrap();
            black_box(&t);
        })
    }

    #[bench]
    fn bench_symmetric(b: &mut Bencher) {
        let opts = QROptions {
            algorithm: QRAlgorithm::Symmetric,
            ..DEFAULT_OPTS
        };

        b.iter(|| {
            let mut a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            for i in 0..BENCH_SIZE {
                for j in 0..i {
                    a[[i, j]] = a[[j, i]]
                }
            }

            let (t, _) = schur_form_opts(a.view(), &opts).unwrap();
            black_box(&t);
        })
    }

    #[bench]
    fn bench_svd(b: &mut Bencher) {
        b.iter(|| {
        let a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            let (_, s, _) = svd(a.view()).unwrap();
            black_box(&s);
        })
    }

    #[bench]
    fn bench_eigenvalues(b: &mut Bencher) {
        b.iter(|| {
            let a = Array::random([BENCH_SIZE, BENCH_SIZE], Uniform::new(-10., 10.));
            let e = eigenvalues(a.view()).unwrap();
            black_box(&e);
        }
        )
    }
}
