use crate::types::*;
use ndarray::array;
use ndarray::s;

use std::cmp::max;
use std::cmp::min;

const EPS: f64 = 1e-4;
const MAX_ITERATIONS: usize = 1000000;

fn norm(v: VectorView) -> f64 {
    v.dot(&v).sqrt()
}

fn proj(v: VectorView, u: VectorView) -> Vector {
    let vu = v.dot(&u);
    let uu = u.dot(&u);
    vu / uu * u.into_owned()
}

pub fn qr_gs(matrix: MatrixView) -> (Matrix, Matrix) {
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

pub fn qr_unshifted(mut m: MatrixViewMut, iterations: usize) -> Matrix {
    let n = m.shape()[0];
    let mut u = Matrix::eye(n);

    for _ in 0..iterations {
        let (q, r) = qr_gs(m.view());
        m.assign(&r.dot(&q));
        u = u.dot(&q);
    }
    u
}

fn givens(a: f64, b: f64) -> (f64, f64) {
    let r = a.hypot(b);
    (a / r, -b / r)
}

fn givens_rot_left(gv: (f64, f64), mut m: MatrixViewMut) {
    let g = array![[gv.0, -gv.1], [gv.1, gv.0]];
    m.assign(&g.dot(&m));
}

fn givens_rot_right(gv: (f64, f64), mut m: MatrixViewMut) {
    let g = array![[gv.0, gv.1], [-gv.1, gv.0]];
    m.assign(&m.dot(&g));
}

pub fn qr_hess(mut m: MatrixViewMut, iterations: usize) -> Matrix {
    let n = m.shape()[0];
    let mut gv = vec![(0., 0.); n - 1];
    let mut u = Matrix::eye(n);

    for _ in 0..iterations {
        for k in 0..n - 1 {
            gv[k] = givens(m[[k, k]], m[[k + 1, k]]);
            givens_rot_left(gv[k], m.slice_mut(s![k..k+2, k..n]));
        }
        for k in 0..n - 1 {
            givens_rot_right(gv[k], m.slice_mut(s![0..k+2, k..k+2]));
            givens_rot_right(gv[k], u.slice_mut(s![0..k+2, k..k+2]));
        }
    }

    u
}

fn hh_vec(x: VectorView) -> Vector {
    let mut u = x.into_owned();
    u[0] += u[0].signum() * norm(x);
    u /= norm(u.view());
    u
}

fn hh_rot_left(u: VectorView, mut m: MatrixViewMut) {
    let u = u.into_shape([u.shape()[0], 1usize]).unwrap();
    m -= &(2. * u.dot(&u.t()).dot(&m));
}

fn hh_rot_right(u: VectorView, mut m: MatrixViewMut) {
    let u = u.into_shape([u.shape()[0], 1usize]).unwrap();
    m -= &(2. * m.dot(&u).dot(&u.t()));
}

pub fn hess_form(mut m: MatrixViewMut) -> Matrix {
    let n = m.shape()[0];
    let mut u = Matrix::eye(n);
    for k in 0..n - 2 {
        let v = hh_vec(m.slice(s![k + 1..n, k]));
        hh_rot_left(v.view(), m.slice_mut(s![k+1..n, k..n]));
        hh_rot_right(v.view(), m.slice_mut(s![0..n, k+1..n]));
        hh_rot_right(v.view(), u.slice_mut(s![0..n, k+1..n]));
    }
    u
}

pub fn qr_francis_shift(mut m: MatrixViewMut, iterations: usize) -> Matrix {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut u = Matrix::eye(n);
    let mut i = 0;

    while p > 1 && i < iterations {
        let q = p - 1;

        let s = m[[q, q]] + m[[p, p]];
        let t = m[[q, q]] * m[[p, p]] - m[[q, p]] * m[[p, q]];

        let mut x = m[[0, 0]] * m[[0, 0]] + m[[0, 1]] * m[[1, 0]] - s * m[[1, 1]] + t;
        let mut y = m[[1, 0]] * (m[[0, 0]] + m[[1, 1]] - s);
        let mut z = m[[1, 0]] * m[[2, 1]];

        for k in 0..p - 1 {
            let v = hh_vec(array![x, y, z].view());
            let r = max(1, k) - 1;
            hh_rot_left(v.view(), m.slice_mut(s![k..k+3, r..n]));

            let r = min(k + 4, p) - 1;
            hh_rot_right(v.view(), m.slice_mut(s![0..r+1, k..k+3]));
            hh_rot_right(v.view(), u.slice_mut(s![0..r+1, k..k+3]));

            if k != p - 2 {
                x = m[[k + 1, k]];
                y = m[[k + 2, k]];
                z = m[[k + 3, k]];
            }
        }

        let gv = givens(x, y);
        givens_rot_left(gv, m.slice_mut(s![q..q+2, p-2..n]));
        givens_rot_right(gv, m.slice_mut(s![0..p+1, q..q+2]));
        givens_rot_right(gv, u.slice_mut(s![0..p+1, q..q+2]));

        if m[[p, q]].abs() < EPS * (m[[q, q]].abs() + m[[p, p]].abs()) {
            m[[p, q]] = 0.;
            p -= 1;
        } else if m[[p - 1, q - 1]].abs() < EPS * (m[[q - 1, q - 1]].abs() + m[[q, q]].abs()) {
            m[[p - 1, q - 1]] = 0.;
            p -= 2;
        }

        i += 1;
    }

    u
}

pub fn schur_decomposition(mut m: MatrixViewMut) -> Matrix {
    let q1 = hess_form(m.view_mut());
    let q2 = qr_francis_shift(m.view_mut(), MAX_ITERATIONS);
    q2.dot(&q1)
}
