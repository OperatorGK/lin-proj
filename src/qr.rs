use crate::types::*;
use ndarray::array;
use ndarray::s;

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
        col /= col.dot(&col).sqrt();
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
    let mut c = vec![0.; n - 1];
    let mut s = vec![0.; n - 1];

    for k in 0..n - 1 {
        let (ck, sk) = givens(matrix[[k, k]], matrix[[k + 1, k]]);
        c[k] = ck;
        s[k] = sk;
        let g = array![[c[k], -s[k]], [s[k], c[k]]];
        let mut slice = matrix.slice_mut(s![k..k+2, k..n]);
        slice.assign(&g.dot(&slice));
    }
    for k in 0..n - 1 {
        let g = array![[c[k], s[k]], [-s[k], c[k]]];
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
