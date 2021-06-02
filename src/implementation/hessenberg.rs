use crate::*;
use crate::implementation::common::*;
use crate::implementation::householder::*;
use crate::implementation::givens::*;

pub fn reduce_to_hessenberg_form(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    for k in 0..n - 2 {
        let v = householder_vec(m.slice(s![k + 1..n, k]));
        householder_refl_left(v.view(), m.slice_mut(s![k + 1..n, k..n]));
        householder_refl_right(v.view(), m.slice_mut(s![0..n, k + 1..n]));
        householder_refl_right(v.view(), u.slice_mut(s![0..n, k + 1..n]));
    }
}

pub fn qr_algorithm_hessenberg(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut gv = vec![(0., 0.); n - 1];

    for _ in 0..opts.iterations {
        if subdiag_convergence(m.view(), opts.eps) {
            break;
        }

        for k in 0..n - 1 {
            gv[k] = givens(m[[k, k]], m[[k + 1, k]]);
            givens_rot_left(gv[k], m.slice_mut(s![k..k + 2, k..n]));
        }
        for k in 0..n - 1 {
            givens_rot_right(gv[k], m.slice_mut(s![0..k + 2, k..k + 2]));
            givens_rot_right(gv[k], u.slice_mut(s![0..n, k..k + 2]));
        }
    }
}