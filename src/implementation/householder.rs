use crate::implementation::common::*;
use crate::*;

#[inline]
pub fn householder_vec(x: VectorView) -> Vector {
    let mut u = x.into_owned();
    u[0] += u[0].signum() * norm(x);
    let n = norm(u.view());
    if n != 0. {
        u /= n;
    };
    u
}

#[inline]
pub fn householder_refl_left(u: VectorView, mut m: MatrixViewMut) {
    let u = u.into_shape([u.shape()[0], 1usize]).unwrap();
    m -= &(2. * u.dot(&u.t()).dot(&m));
}

#[inline]
pub fn householder_refl_right(u: VectorView, mut m: MatrixViewMut) {
    let u = u.into_shape([u.shape()[0], 1usize]).unwrap();
    m -= &(2. * m.dot(&u).dot(&u.t()));
}
