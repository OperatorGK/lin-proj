use crate::implementation::blocks::*;
use crate::implementation::common::*;
use crate::implementation::givens::*;
use crate::implementation::householder::*;
use crate::*;

use ndarray::{array, s};
use std::cmp::{max, min};

#[inline]
fn francis_reflection_axis(m: MatrixView, p: usize) -> Vector {
    let q = p - 1;
    let trace = m[[q, q]] + m[[p, p]];
    let det = m[[q, q]] * m[[p, p]] - m[[q, p]] * m[[p, q]];

    let x = m[[0, 0]] * m[[0, 0]] + m[[0, 1]] * m[[1, 0]] - trace * m[[0, 0]] + det;
    let y = m[[1, 0]] * (m[[0, 0]] + m[[1, 1]] - trace);
    let z = m[[1, 0]] * m[[2, 1]];

    array![x, y, z]
}

fn francis_qr_step(mut m: MatrixViewMut, mut u: MatrixViewMut, mut v: Vector, p: usize, acc: bool) {
    let n = m.shape()[0];
    for k in 0..p - 1 {
        let refl = householder_vec(v.view());

        let r = max(1, k);
        householder_refl_left(refl.view(), m.slice_mut(s![k..k + 3, r - 1..n]));

        let r = min(k + 4, p + 1);
        householder_refl_right(refl.view(), m.slice_mut(s![0..r, k..k + 3]));

        if acc {
            householder_refl_right(refl.view(), u.slice_mut(s![0..n, k..k + 3]));
        }

        v[0] = m[[k + 1, k]];
        v[1] = m[[k + 2, k]];
        if k != p - 2 {
            v[2] = m[[k + 3, k]];
        }
    }

    let rot = givens(v[0], v[1]);
    givens_rot_left(rot, m.slice_mut(s![p - 1..p + 1, p - 2..n]));
    givens_rot_right(rot, m.slice_mut(s![0..p + 1, p - 1..p + 1]));

    if acc {
        givens_rot_right(rot, u.slice_mut(s![0..n, p - 1..p + 1]));
    }
}

pub fn qr_algorithm_francis(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut i = 0;

    while p > 1 && i < opts.iterations {
        let q = p - 1;
        if eigval_collapsed(opts.eps, m[[p, q]], m[[q, q]], m[[p, p]]) {
            m[[p, q]] = 0.;
            p -= 1;
            continue;
        } else if eigval_collapsed(opts.eps, m[[p - 1, q - 1]], m[[q - 1, q - 1]], m[[q, q]]) {
            m[[p - 1, q - 1]] = 0.;
            p -= 2;
            continue;
        }

        let v = francis_reflection_axis(m.view(), p);
        francis_qr_step(
            m.view_mut(),
            u.view_mut(),
            v,
            p,
            opts.accumulate_sim_transforms,
        );

        i += 1;
    }
}

pub fn francis_block_reduction(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let (mut i, n) = (0, m.shape()[0]);
    while i + 1 < n {
        if m[[i + 1, i]].abs() < opts.eps {
            i += 1;
            continue;
        }
        let rot = rot_2by2(m.slice(s![i..i + 2, i..i + 2]));
        givens_rot_left(rot, m.slice_mut(s![i..i + 2, i..n]));
        givens_rot_right(rot, m.slice_mut(s![0..i + 2, i..i + 2]));

        if opts.accumulate_sim_transforms {
            givens_rot_right(rot, u.slice_mut(s![0..n, i..i + 2]));
        }
        i += 2;
    }
}
