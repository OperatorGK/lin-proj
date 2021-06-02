use crate::*;
use crate::implementation::common::*;
use crate::implementation::hessenberg::{hessenberg_form};
use crate::implementation::qr_symmetric::*;

use ndarray::Axis;

pub fn svd(m: MatrixView, opts: &QROptions) -> (Matrix, Vector, Matrix) {
    let mut s = m.dot(&m.t());
    let mut u = Matrix::eye(s.shape()[0]);
    hessenberg_form(s.view_mut(), u.view_mut(), opts);
    qr_algorithm_symmetric(s.view_mut(), u.view_mut(), opts);

    let mut z: Vector = s.diag().into_iter().map(|x| x.sqrt()).collect();
    sort_diagonal_values(z.view_mut(), u.view_mut());

    let mut vs: Vec<Vector> = (0..z.shape()[0])
        .map(|i| m.t().dot(&u.column(i)) / z[i])
        .collect();

    let mut eyes: Vec<Vector> = Matrix::eye(m.shape()[1])
        .gencolumns()
        .into_iter()
        .map(|x| x.into_owned())
        .collect();

    vs.append(&mut eyes);
    orthonormalize(vs.as_mut_slice());
    let vt: Matrix = stack_owned(Axis(0), &vs[0..m.shape()[1]]);
    (u, z, vt)
}
