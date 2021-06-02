use crate::*;
use crate::implementation::common::*;

use ndarray::Axis;

pub fn qr_decomposition(m: MatrixView) -> (Matrix, Matrix) {
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
        let (q, r) = qr_decomposition(m.view());
        m.assign(&r.dot(&q));
        if opts.accumulate_sim_transforms {
            u.assign(&u.dot(&q));
        }
    }
}
