use crate::*;

#[inline]
pub fn rot_2by2_real(m: MatrixView, e: f64) -> (f64, f64) {
    let v = match (m[[0, 1]], m[[1, 0]]) {
        (b, _) if b != 0. => (b, e - m[[0, 0]]),
        (_, c) if c != 0. => (e - m[[1, 1]], c),
        (_, _) => (1., 0.),
    };
    let h = v.0.hypot(v.1);
    (v.0 / h, -v.1 / h)
}

#[inline]
pub fn rot_2by2_complex(m: MatrixView) -> (f64, f64) {
    let a = m[[0, 1]] + m[[1, 0]];
    let b = m[[0, 0]] - m[[1, 1]];
    let phi = b.atan2(a) / 2.;
    (phi.cos(), phi.sin())
}

#[inline]
pub fn rot_2by2(m: MatrixView) -> (f64, f64) {
    let p = m[[0, 0]] + m[[1, 1]]; // x^2 - px + q
    let q = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
    let d = p * p - 4. * q;
    if d < 0. {
        rot_2by2_complex(m)
    } else {
        rot_2by2_real(m, (d.sqrt() + p) / 2.)
    }
}
