use crate::implementation::blocks::*;
use crate::implementation::common::*;
use crate::implementation::givens::*;
use crate::*;

use ndarray::s;

#[inline]
fn wilkinson_shift(m: MatrixView, p: usize) -> f64 {
    let a = m[[p, p]];
    let b = m[[p, p - 1]];
    let d = 0.5 * (m[[p - 1, p - 1]] - a);
    if d == 0. {
        a - b.abs()
    } else {
        a - b * b / (d + d.signum() * d.hypot(b))
    }
}

#[inline]
fn implicit_tridiagonal_rotation(
    mut m: MatrixViewMut,
    x: &mut f64,
    y: &mut f64,
    c: f64,
    s: f64,
    k: usize,
    p: usize,
) {
    let w = c * *x - s * *y;
    let d = m[[k, k]] - m[[k + 1, k + 1]];
    let z = (2. * c * m[[k + 1, k]] + d * s) * s;

    m[[k, k]] -= z;
    m[[k + 1, k + 1]] += z;
    m[[k + 1, k]] = d * c * s + (c * c - s * s) * m[[k + 1, k]];
    m[[k, k + 1]] = m[[k + 1, k]];
    *x = m[[k + 1, k]];

    if k > 0 {
        m[[k - 1, k]] = w;
        m[[k, k - 1]] = w;
    }

    if k + 1 < p {
        *y = -s * m[[k + 2, k + 1]];
        m[[k + 2, k + 1]] *= c;
        m[[k + 1, k + 2]] *= c;
    }
}

fn symmetric_qr_step(mut m: MatrixViewMut, mut u: MatrixViewMut, p: usize, s: f64) {
    let n = m.shape()[0];
    let mut x = m[[0, 0]] - s;
    let mut y = m[[0, 1]];

    for k in 0..p {
        let (c, s) = if p > 1 {
            givens(x, y)
        } else {
            rot_2by2(m.slice(s![0..2, 0..2]))
        };

        implicit_tridiagonal_rotation(m.view_mut(), &mut x, &mut y, c, s, k, p);
        givens_rot_right((c, s), u.slice_mut(s![0..n, k..k + 2]));
    }
}

pub fn qr_algorithm_symmetric(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut i = 0;

    while p > 0 && i < opts.iterations {
        let s = wilkinson_shift(m.view(), p);
        symmetric_qr_step(m.view_mut(), u.view_mut(), p, s);
        if eigval_collapsed(opts.eps, m[[p, p - 1]], m[[p - 1, p - 1]], m[[p, p]]) {
            p -= 1;
        }
        i += 1;
    }
}
