use crate::*;
use ndarray::{Axis, stack};
use std::cmp::Ordering;

#[inline]
pub fn norm(v: VectorView) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[inline]
pub fn proj(v: VectorView, u: VectorView) -> Vector {
    let vu = v.dot(&u);
    let uu = u.dot(&u);
    vu / uu * u.into_owned()
}

#[inline]
pub fn stack_owned(axis: Axis, vs: &[Vector]) -> Matrix {
    let views: Vec<VectorView> = vs.into_iter().map(|x| x.view()).collect();
    stack(axis, views.as_slice()).unwrap()
}

#[inline]
pub fn eigval_collapsed(eps: f64, subdiag: f64, upper: f64, lower: f64) -> bool {
    subdiag.abs() < eps * (upper.abs() + lower.abs())
}

pub fn sort_diagonal_values(mut z: VectorViewMut, mut u: MatrixViewMut) {
    let mut perm: Vec<usize> = (0..z.shape()[0]).collect();
    perm.sort_by(|i1, i2| z[[*i2]].partial_cmp(&z[[*i1]]).unwrap_or(Ordering::Equal));
    let new_z: Vector = (&perm).into_iter().map(|i| z[[*i]]).collect();
    let cols: Vec<VectorView> = perm.into_iter().map(|i| u.column(i)).collect();
    let new_u: Matrix = stack(Axis(1), cols.as_slice()).unwrap();
    z.assign(&new_z);
    u.assign(&new_u);
}

pub fn orthonormalize(orth: &mut [Vector]) {
    let n = orth.len();
    for i in 0..n {
        for j in 0..i {
            orth[i] -= &proj(orth[i].view(), orth[j].view());
        }
        orth[i] /= norm(orth[i].view());
    }
}

pub fn zero_subeps_entries(mut m: MatrixViewMut, eps: f64) {
    m.map_inplace(|x| if x.abs() < eps { *x = 0.; })
}

pub fn extract_eigenvalues(m: MatrixView) -> Vec<Complex> {
    let (mut i, n) = (0, m.shape()[0]);
    let mut eigs = Vec::new();

    while i + 1 < n {
        let re = m[[i, i]];
        if m[[i + 1, i]] == 0. {
            eigs.push(Complex::from(re));
            i += 1;
            continue;
        }

        let im = (m[[i + 1, i]] * m[[i, i + 1]]).sqrt();
        eigs.push(Complex::new(re, im));
        eigs.push(Complex::new(re, -im));
        i += 2;
    }

    eigs
}
