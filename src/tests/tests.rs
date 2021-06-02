#![cfg(test)]

use crate::implementation::checks::*;
use crate::*;

use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::cmp::min;

const EPS: f64 = 1e-4;

fn random_check_qr(sz: usize) {
    let a = Array::random([sz, sz], Uniform::new(-10., 10.));
    let (q, r) = qr_decomposition(a.view()).unwrap();

    assert!(diff_unit(q.view()) < EPS);
    assert!(diff_triag(r.view()) < EPS);
    assert!(diff_rel(a.view(), q.dot(&r).view()) < EPS);
}

#[test]
fn test_gram_schmidt() {
    for sz in [1, 2, 3, 5, 10, 20, 50] {
        for _ in 0..10 {
            random_check_qr(sz);
        }
    }
}

fn random_check_hess(sz: usize) {
    let a = Array::random([sz, sz], Uniform::new(-10., 10.));
    let (t, u) = hessenberg_form(a.view()).unwrap();

    assert!(diff_subtriag(t.view()) < EPS);
    assert!(diff_unit(u.view()) < EPS);
    assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
}

#[test]
fn test_hess_form() {
    for sz in [1, 2, 3, 5, 10, 20, 50] {
        for _ in 0..10 {
            random_check_hess(sz);
        }
    }
}

fn random_check_svd(sz1: usize, sz2: usize) {
    let a = Array::random([sz1, sz2], Uniform::new(-10., 10.));
    let (u, s, vt) = svd(a.view()).unwrap();
    let mut ss = crate::Matrix::zeros((sz1, sz2));
    for i in 0..min(sz1, sz2) {
        ss[[i, i]] = s[i];
    }

    // Low accuracy
    assert!(diff_unit(u.view()) < 0.01);
    assert!(diff_unit(vt.view()) < 0.01);
}

#[test]
fn test_svd() {
    for sz1 in [1, 2, 3, 10] {
        for sz2 in [1, 2, 3, 10] {
            random_check_svd(sz1, sz2);
        }
    }
}
