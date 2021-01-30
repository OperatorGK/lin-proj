use crate::types::*;
use ndarray::array;
use ndarray::s;

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

pub fn qr_unshifted(matrix: MatrixView, iterations: usize) -> Vector {
    let mut a = matrix.into_owned();
    for _ in 0..iterations {
        let (q, r) = qr_gs(a.view());
        a = r.dot(&q);
    }
    a.into_diag()
}

pub fn givens(a: f64, b: f64) -> (f64, f64) {
    let r = a.hypot(b);
    (a / r, -b / r)
}

pub fn qr_hess_step(mut matrix: MatrixViewMut) {
    let n = matrix.shape()[0];
    let mut gv = vec![(0., 0.); n - 1];

    for k in 0..n - 1 {
        gv[k] = givens(matrix[[k, k]], matrix[[k + 1, k]]);
        let g = array![[gv[k].0, -gv[k].1], [gv[k].1, gv[k].0]];
        let mut slice = matrix.slice_mut(s![k..k+2, k..n]);
        slice.assign(&g.dot(&slice));
    }
    for k in 0..n - 1 {
        let g = array![[gv[k].0, gv[k].1], [-gv[k].1, gv[k].0]];
        let mut slice = matrix.slice_mut(s![0..k+2, k..k+2]);
        slice.assign(&slice.dot(&g));
    }
}

pub fn qr_hess(matrix: MatrixView, iterations: usize) -> Vector {
    let mut a = matrix.into_owned();
    for _ in 0..iterations {
        qr_hess_step(a.view_mut());
    }
    a.into_diag()
}

pub fn hh_vec(x: VectorView) -> Vector {
    let mut u = x.into_owned();
    u[0] += u[0].signum() * norm(x);
    u /= norm(u.view());
    u
}

pub fn hess_form(mut matrix: MatrixViewMut) {
    let n = matrix.shape()[0];
    for k in 0..n - 2 {
        let u = hh_vec(matrix.slice(s![k + 1..n, k])).into_shape([n - k - 1, 1usize]).unwrap();
        let mut slice = matrix.slice_mut(s![k+1..n, k..n]);
        slice -= &(2. * u.dot(&u.t()).dot(&slice));
        let mut slice = matrix.slice_mut(s![0..n, k+1..n]);
        slice -= &(2. * slice.dot(&u).dot(&u.t()));
    }
}
