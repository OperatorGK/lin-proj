use crate::types::*;

fn proj(v: VectorView, u: VectorView) -> Vector {
    let vu = v.dot(&u);
    let uu = u.dot(&u);
    vu / uu * u.into_owned()
}

pub fn qr_gs(matrix: MatrixView) -> (Matrix, Matrix) {
    let n = matrix.shape()[0];
    let mut q = Matrix::zeros((n, n));

    for i in 0..n {
        let mut col = matrix.column(i).into_owned();
        for j in 0..i {
            col -= &proj(col.view(), q.column(j));
        }
        col /= col.dot(&col).sqrt();
        q.column_mut(i).assign(&col);
    }

    let mut r = Matrix::zeros((n, n));

    for i in 0..n {
        for j in 0..i + 1 {
            r[[j, i]] = q.column(j).dot(&matrix.column(i));
        }
    }

    (q, r)
}
