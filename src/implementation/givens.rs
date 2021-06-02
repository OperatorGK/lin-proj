use crate::*;

#[inline]
pub fn givens(a: f64, b: f64) -> (f64, f64) {
    if b != 0. {
        let r = a.hypot(b);
        (a / r, -b / r)
    } else {
        (1., 0.)
    }
}

#[inline]
pub fn givens_rot_left(gv: (f64, f64), mut m: MatrixViewMut) {
    let g = array![[gv.0, -gv.1], [gv.1, gv.0]];
    m.assign(&g.dot(&m));
}

#[inline]
pub fn givens_rot_right(gv: (f64, f64), mut m: MatrixViewMut) {
    let g = array![[gv.0, gv.1], [-gv.1, gv.0]];
    m.assign(&m.dot(&g));
}

