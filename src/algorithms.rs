use crate::types::*;
use ndarray::{array, s, stack, Axis};

use std::cmp::{max, min, Ordering};

pub const DEFAULT_OPTS: QROptions =
    QROptions {
        eps: 1e-8,
        iterations: 1_000,
        algorithm: QRAlgorithm::Default,
    };

#[inline]
fn norm(v: VectorView) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[inline]
fn proj(v: VectorView, u: VectorView) -> Vector {
    let vu = v.dot(&u);
    let uu = u.dot(&u);
    vu / uu * u.into_owned()
}

#[inline]
fn stack_owned(axis: Axis, vs: &[Vector]) -> Matrix {
    let views: Vec<VectorView> = vs.into_iter().map(|x| x.view()).collect();
    stack(axis, views.as_slice()).unwrap()
}

fn orthonormalize(orth: &mut [Vector]) {
    let n = orth.len();
    for i in 0..n {
        for j in 0..i {
            orth[i] -= &proj(orth[i].view(), orth[j].view());
        }
        orth[i] /= norm(orth[i].view());
    }
}

pub fn qr_decompose_gram_schmidt(m: MatrixView) -> (Matrix, Matrix) {
    let n = m.shape()[0];

    let mut qv: Vec<Vector> = m.gencolumns().into_iter().map(|x| x.into_owned()).collect();
    orthonormalize(qv.as_mut_slice());
    let q: Matrix = stack_owned(Axis(1), qv.as_slice());

    let mut r = Matrix::zeros((n, n));
    for i in 0..n {
        for j in 0..i + 1 {
            r[[j, i]] = q.column(j).dot(&m.column(i));
        }
    }

    (q, r)
}

pub fn qr_algorithm_naive(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    for _ in 0..opts.iterations {
        let (q, r) = qr_decompose_gram_schmidt(m.view());
        m.assign(&r.dot(&q));
        u.assign(&u.dot(&q));
    }
}

#[inline]
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b != 0. {
        let r = a.hypot(b);
        (a / r, -b / r)
    } else {
        (1., 0.)
    }
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
            givens_rot_right(gv[k], u.slice_mut(s![0..n, k..k+2]));
        }
    }
}

#[inline]
fn householder_vec(x: VectorView) -> Vector {
    let mut u = x.into_owned();
    u[0] += u[0].signum() * norm(x);
    let n = norm(u.view());
    if n != 0. {
        u /= n;
    };
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

#[inline]
fn rot_2by2_real(m: MatrixView, e: f64) -> (f64, f64) {
    let v = match (m[[0, 1]], m[[1, 0]]) {
        (b, _) if b != 0. => (b, e - m[[0, 0]]),
        (_, c) if c != 0. => (e - m[[1, 1]], c),
        (_, _) => (1., 0.),
    };
    let h = v.0.hypot(v.1);
    (v.0 / h, -v.1 / h)
}

#[inline]
fn rot_2by2_complex(m: MatrixView) -> (f64, f64) {
    let a = m[[0, 1]] + m[[1, 0]];
    let b = m[[0, 0]] - m[[1, 1]];
    let phi = b.atan2(a) / 2.;
    (phi.cos(), phi.sin())
}

#[inline]
fn rot_2by2(m: MatrixView) -> (f64, f64) {
    let p = m[[0, 0]] + m[[1, 1]];      // x^2 - px + q
    let q = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    let d = p * p - 4. * q;
    if d < 0. {
        rot_2by2_complex(m)
    } else {
        rot_2by2_real(m, (d.sqrt() + p) / 2.)
    }
}

#[inline]
fn eigval_collapsed(eps: f64, subdiag: f64, upper: f64, lower: f64) -> bool {
    subdiag.abs() < eps * (upper.abs() + lower.abs())
}

fn implicit_double_step(mut m: MatrixViewMut, mut u: MatrixViewMut, mut v: VectorViewMut, p: usize) {
    let n = m.shape()[0];
    for k in 0..p - 1 {
        let refl = householder_vec(v.view());

        let r = max(1, k);
        householder_refl_left(refl.view(), m.slice_mut(s![k..k+3, r-1..n]));

        let r = min(k + 4, p);
        householder_refl_right(refl.view(), m.slice_mut(s![0..r, k..k+3]));
        householder_refl_right(refl.view(), u.slice_mut(s![0..n, k..k+3]));

        v[0] = m[[k + 1, k]];
        v[1] = m[[k + 2, k]];
        if k != p - 2 {
            v[2] = m[[k + 3, k]];
        }
    }

    let rot = givens(v[0], v[1]);
    givens_rot_left(rot, m.slice_mut(s![p-1..p+1, p-2..n]));
    givens_rot_right(rot, m.slice_mut(s![0..p+1, p-1..p+1]));
    givens_rot_right(rot, u.slice_mut(s![0..n, p-1..p+1]));
}

pub fn qr_algorithm_francis(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut i = 0;

    while p > 1 && i < opts.iterations {
        let q = p - 1;

        let s = m[[q, q]] + m[[p, p]];
        let t = m[[q, q]] * m[[p, p]] - m[[q, p]] * m[[p, q]];

        let x = m[[0, 0]] * m[[0, 0]] + m[[0, 1]] * m[[1, 0]] - s * m[[0, 0]] + t;
        let y = m[[1, 0]] * (m[[0, 0]] + m[[1, 1]] - s);
        let z = m[[1, 0]] * m[[2, 1]];

        let mut v = array![x, y, z];
        implicit_double_step(m.view_mut(), u.view_mut(), v.view_mut(), p);

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

pub fn francis_block_reduction(mut m: MatrixViewMut, mut u: MatrixViewMut, eps: f64) {
    let (mut i, n) = (0, m.shape()[0]);
    while i + 1 < n {
        if m[[i + 1, i]].abs() < eps {
            i += 1;
            continue;
        }
        let rot = rot_2by2(m.slice(s![i..i+2, i..i+2]));
        givens_rot_left(rot, m.slice_mut(s![i..i+2, i..n]));
        givens_rot_right(rot, m.slice_mut(s![0..i+2, i..i+2]));
        givens_rot_right(rot, u.slice_mut(s![0..n, i..i+2]));
        i += 2;
    }
}

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

fn implicit_symmetric_qr_step(mut m: MatrixViewMut, mut u: MatrixViewMut, p: usize, s: f64) {
    let n = m.shape()[0];
    let mut x = m[[0, 0]] - s;
    let mut y = m[[0, 1]];

    for k in 0..p {
        let (c, s) = if p > 1 {
            givens(x, y)
        } else {
            rot_2by2(m.slice(s![0..2, 0..2]))
        };
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

        if k + 1 < p {
            y = -s * m[[k + 2, k + 1]];
            m[[k + 2, k + 1]] *= c;
            m[[k + 1, k + 2]] *= c;
        }

        givens_rot_right((c, s), u.slice_mut(s![0..n, k..k+2]));
    }
}

pub fn symmetric_qr_algorithm(mut m: MatrixViewMut, mut u: MatrixViewMut, opts: &QROptions) {
    let n = m.shape()[0];
    let mut p = n - 1;
    let mut i = 0;

    while p > 0 && i < opts.iterations {
        let s = wilkinson_shift(m.view(), p);
        implicit_symmetric_qr_step(m.view_mut(), u.view_mut(), p, s);
        if eigval_collapsed(opts.eps, m[[p, p - 1]], m[[p - 1, p - 1]], m[[p, p]]) {
            p -= 1;
        }
        i += 1;
    }
}

fn sort_singular_values(mut z: VectorViewMut, mut u: MatrixViewMut) {
    let mut perm: Vec<usize> = (0..z.shape()[0]).collect();
    perm.sort_by(|i1, i2| z[[*i2]].partial_cmp(&z[[*i1]]).unwrap_or(Ordering::Equal));
    let new_z: Vector = (&perm).into_iter().map(|i| z[[*i]]).collect();
    let cols: Vec<VectorView> = perm.into_iter().map(|i| u.column(i)).collect();
    let new_u: Matrix = stack(Axis(1), cols.as_slice()).unwrap();
    z.assign(&new_z);
    u.assign(&new_u);
}

pub fn svd(m: MatrixView) -> (Matrix, Vector, Matrix) {
    let mut s = m.dot(&m.t());
    let mut u = Matrix::eye(s.shape()[0]);
    reduce_to_hessenberg_form(s.view_mut(), u.view_mut());
    symmetric_qr_algorithm(s.view_mut(), u.view_mut(), &DEFAULT_OPTS);
    let mut z: Vector = s.diag().into_iter().map(|x| x.sqrt()).collect();
    sort_singular_values(z.view_mut(), u.view_mut());
    let mut vs: Vec<Vector> = (0..z.shape()[0]).map(|i| m.t().dot(&u.column(i)) / z[i]).collect();
    let mut eyes: Vec<Vector> = Matrix::eye(m.shape()[1]).gencolumns().into_iter().map(|x| x.into_owned()).collect();
    vs.append(&mut eyes);
    orthonormalize(vs.as_mut_slice());
    let vt: Matrix = stack_owned(Axis(0), &vs[0..m.shape()[1]]);
    (u, z, vt)
}
