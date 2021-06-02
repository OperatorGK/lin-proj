#![cfg(test)]

use crate::implementation::checks::*;
use crate::*;

use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

const OPTS: QROptions = QROptions {
    iterations: 1000,
    ..DEFAULT_OPTS
};
const EPS: f64 = 1e-4;

#[test]
fn test_qr_naive() {
    let a = ndarray::array![[2., -3.], [-1., 1.]];
    let opts = QROptions {
        algorithm: QRAlgorithm::Naive,
        ..OPTS
    };
    let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

    assert!(diff_triag(t.view()) < EPS);
    assert!(diff_unit(u.view()) < EPS);
    assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
}

#[test]
fn test_qr_hess() {
    let a = ndarray::array![[3., 7., 5.], [-1., -3., -4.], [0., 1., -9.]];
    let opts = QROptions {
        algorithm: QRAlgorithm::Hessenberg,
        ..OPTS
    };
    let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

    assert!(diff_triag(t.view()) < EPS);
    assert!(diff_unit(u.view()) < EPS);
    assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
}

fn random_check_qr_francis(sz: usize) {
    let a = Array::random([sz, sz], Uniform::new(-10., 10.));
    let opts = QROptions {
        algorithm: QRAlgorithm::Francis,
        ..OPTS
    };
    let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

    assert!(diff_subtriag(t.view()) < EPS);
    assert!(diff_unit(u.view()) < EPS);
    assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
}

#[test]
fn test_qr_francis() {
    for sz in [1, 2, 3, 5, 10, 20, 50] {
        for _ in 0..10 {
            random_check_qr_francis(sz);
        }
    }
}

fn random_check_qr_symmetric(sz: usize) {
    let mut a = Array::random([sz, sz], Uniform::new(-10., 10.));
    for i in 0..sz {
        for j in 0..i {
            a[[i, j]] = a[[j, i]]
        }
    }

    let opts = QROptions {
        algorithm: QRAlgorithm::Symmetric,
        ..OPTS
    };
    let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

    assert!(diff_triag(t.view()) < EPS);
    assert!(diff_unit(u.view()) < EPS);
    assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
}

#[test]
fn test_qr_symmetric() {
    for sz in [1, 2, 3, 5, 10, 20, 50] {
        for _ in 0..10 {
            random_check_qr_symmetric(sz);
        }
    }
}
