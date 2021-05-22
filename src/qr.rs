use crate::types::*;
use ndarray::array;
use ndarray::s;

use std::cmp::max;
use std::cmp::min;

pub const DEFAULT_OPTS: QROptions =
    QROptions {
        eps: 1e-8,
        iterations: 1_000,
    };

#[inline]
fn norm(v: VectorView) -> f64 {
    v.dot(&v).sqrt()
}

#[inline]
fn proj(v: VectorView, u: VectorView) -> Vector {
    let vu = v.dot(&u);
    let uu = u.dot(&u);
    vu / uu * u.into_owned()
}

pub fn qr_decompose_gram_schmidt(matrix: MatrixView) -> (Matrix, Matrix) {
    let n = matrix.shape()[0];

    let mut q = Matrix::zeros((n, n));
    for i in 0..n {
        let mut col = matrix.column(i).into_owned();
        for j in 0..i {
            col -= &proj(col.view(), q.column(j));
        }
        col /= norm(col.view());
        q.column_mut(i).assign(&col);
    }

    let mut r = Matrix::zeros((n, n));
    for i in 0..n {
        for j in 0..i + 1 {
            r[[j, i]] = q.column(j).dot(&matrix.column(i));
        }
    }

    (q, r)
}

pub fn qr_algorithm_naive(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    for _ in 0..opts.iterations {
        let (q, r) = qr_decompose_gram_schmidt(m.view());
        m.assign(&r.dot(&q));
        u = u.dot(&q);
    }
}

#[inline]
fn givens(a: f64, b: f64) -> (f64, f64) {
    let r = a.hypot(b);
    (a / r, -b / r)
}

#[inline]
fn givens_rot_left(gv: (f64, f64), mut m: MatrixViewMut) {
    let g = array![[gv.0, -gv.1], [gv.1, gv.0]];
    m.assign(&g.dot(&m));
}

#[inline]
fn givens_rot_right(gv: (f64, f64), mut m: MatrixViewMut) {
    let g = array![[gv.0, gv.1], [-gv.1, gv.0]];
    m.assign(&m.dot(&g));
}

pub fn qr_algorithm_hessenberg(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut gv = vec![(0., 0.); n - 1];

    for _ in 0..opts.iterations {
        for k in 0..n - 1 {
            gv[k] = givens(m[[k, k]], m[[k + 1, k]]);
            givens_rot_left(gv[k], m.slice_mut(s![k..k+2, k..n]));
        }
        for k in 0..n - 1 {
            givens_rot_right(gv[k], m.slice_mut(s![0..k+2, k..k+2]));
            givens_rot_right(gv[k], u.slice_mut(s![0..k+2, k..k+2]));
        }
    }
}

#[inline]
fn householder_vec(x: VectorView) -> Vector {
    let mut u = x.into_owned();
    u[0] += u[0].signum() * norm(x);
    u /= norm(u.view());
    u
}

#[inline]
fn householder_refl_left(u: VectorView, mut m: MatrixViewMut) {
    let u = u.into_shape([u.shape()[0], 1usize]).unwrap();
    m -= &(2. * u.dot(&u.t()).dot(&m));
}

#[inline]
fn householder_refl_right(u: VectorView, mut m: MatrixViewMut) {
    let u = u.into_shape([u.shape()[0], 1usize]).unwrap();
    m -= &(2. * m.dot(&u).dot(&u.t()));
}

pub fn reduce_to_hessenberg_form(mut m: MatrixViewMut, mut u: MatrixViewMut) {
    let n = m.shape()[0];
    for k in 0..n - 2 {
        let v = householder_vec(m.slice(s![k + 1..n, k]));
        householder_refl_left(v.view(), m.slice_mut(s![k+1..n, k..n]));
        householder_refl_right(v.view(), m.slice_mut(s![0..n, k+1..n]));
        householder_refl_right(v.view(), u.slice_mut(s![0..n, k+1..n]));
    }
}

pub fn schur_decomposition_2by2(mut m: MatrixViewMut, mut u: MatrixViewMut) {
    let p = m[[0, 0]] + m[[1, 1]];  // x^2 - px + q
    let q = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    let d = p * p - 4. * q;
}

#[inline]
fn eigval_collapsed(eps: f64, subdiag: f64, upper: f64, lower: f64) -> bool {
    subdiag.abs() < eps * (upper.abs() + lower.abs())
}

fn francis_refl(mut m: MatrixViewMut, mut u: MatrixViewMut, mut v: VectorViewMut, p: usize) {
    let n = m.shape()[0];
    for k in 0..p - 1 {
        let refl = householder_vec(v.view());

        let r = max(1, k) - 1;
        householder_refl_left(refl.view(), m.slice_mut(s![k..k+3, r..n]));

        let r = min(k + 4, p) - 1;
        householder_refl_right(refl.view(), m.slice_mut(s![0..r+1, k..k+3]));
        householder_refl_right(refl.view(), u.slice_mut(s![0..r+1, k..k+3]));

        if k != p - 2 {
            v[0] = m[[k + 1, k]];
            v[1] = m[[k + 2, k]];
            v[2] = m[[k + 3, k]];
        }
    }
}

pub fn qr_algorithm_francis(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut i = 0;

    while p > 1 && i < opts.iterations {
        let q = p - 1;

        let s = m[[q, q]] + m[[p, p]];
        let t = m[[q, q]] * m[[p, p]] - m[[q, p]] * m[[p, q]];

        let x = m[[0, 0]] * m[[0, 0]] + m[[0, 1]] * m[[1, 0]] - s * m[[1, 1]] + t;
        let y = m[[1, 0]] * (m[[0, 0]] + m[[1, 1]] - s);
        let z = m[[1, 0]] * m[[2, 1]];

        let mut v = array![x, y, z];
        francis_refl(m.view_mut(), u.view_mut(), v.view_mut(), p);

        let gv = givens(x, y);
        givens_rot_left(gv, m.slice_mut(s![q..q+2, p-2..n]));
        givens_rot_right(gv, m.slice_mut(s![0..p+1, q..q+2]));
        givens_rot_right(gv, u.slice_mut(s![0..p+1, q..q+2]));

        if eigval_collapsed(opts.eps, m[[p, q]], m[[q, q]], m[[p, p]]) {
            m[[p, q]] = 0.;
            p -= 1;
        } else if eigval_collapsed(opts.eps, m[[p - 1, q - 1]], m[[q - 1, q - 1]], m[[q, q]]) {
            m[[p - 1, q - 1]] = 0.;
            p -= 2;
        }

        i += 1;
    }
}

pub fn real_schur_form(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    reduce_to_hessenberg_form(m.view_mut(), u.view_mut());
    qr_algorithm_francis(m.view_mut(), u.view_mut(), opts);
}

#[inline]
fn symmetric_wilkinson_shift(m: MatrixView, p: usize) -> f64 {
    let a = m[[p, p]];
    let b = m[[p, p - 1]];
    let d = 0.5 * (m[[p - 1, p - 1]] - a);
    if d == 0. {
        a - b.abs()
    } else {
        a - b * b / (d + d.signum() * d.hypot(b))
    }
}

fn implicit_symmetric_qr_step(mut m: MatrixViewMut, mut u: MatrixViewMut, p: usize) {
    let mut x = m[[0, 0]];
    let mut y = m[[0, 1]];

    for k in 0..p - 1 {
        let (c, s) = givens(x, y);
        let w = c * x - s * y;
        let d = m[[k, k]] - m[[k + 1, k + 1]];
        let z = (2. * c * m[[k + 1, k]] + d * s) * s;

        m[[k, k]] -= z;
        m[[k + 1, k + 1]] += z;
        m[[k + 1, k]] = d * c * s + (c * c - s * s) * m[[k + 1, k]];
        m[[k, k + 1]] = m[[k + 1, k]];
        x = m[[k + 1, k]];

        if k > 0 {
            m[[k - 1, k]] = w;
            m[[k, k - 1]] = w;
        }

        if k < p - 2 {
            y = -s * m[[k + 2, k + 1]];
            m[[k + 2, k + 1]] *= c;
            m[[k + 1, k + 2]] *= c;
        }

        givens_rot_right((c, s), u.view_mut());
    }
}

pub fn symmetric_qr_algorithm(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut i = 0;

    while p > 0 && i < opts.iterations {
        let s = symmetric_wilkinson_shift(m.view(), p);
        implicit_symmetric_qr_step(m.view_mut(), u.view_mut(), p);
        if eigval_collapsed(opts.eps, m[[p, p - 1]], m[[p - 1, p - 1]], m[[p, p]]) {
            p -= 1;
        }
    }
}
