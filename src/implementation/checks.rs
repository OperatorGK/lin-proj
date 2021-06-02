use crate::types::*;

pub fn frob_norm(m: MatrixView) -> f64 {
    m.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn diff_rel(orig: MatrixView, res: MatrixView) -> f64 {
    let diff = &res - &orig;
    frob_norm(diff.view()) / frob_norm(orig.view())
}

pub fn diff_unit(u: MatrixView) -> f64 {
    diff_rel(u.dot(&u.t()).view(), Matrix::eye(u.shape()[0]).view())
}

pub fn diff_triag(m: MatrixView) -> f64 {
    let n = m.shape()[0];
    let mut t = m.into_owned();
    for i in 0..n {
        for j in 0..i {
            t[[i, j]] = 0.;
        }
    }
    diff_rel(t.view(), m)
}

pub fn diff_subtriag(m: MatrixView) -> f64 {
    let n = m.shape()[0];
    let mut t = m.into_owned();
    for i in 1..n {
        for j in 0..i - 1 {
            t[[i, j]] = 0.;
        }
    }
    diff_rel(t.view(), m)
}

pub fn diff_symm(m: MatrixView) -> f64 {
    let n = m.shape()[0];
    let mut t = m.into_owned();
    for i in 0..n {
        for j in 0..i {
            t[[i, j]] = t[[j, i]];
        }
    }
    diff_rel(t.view(), m)
}

pub fn finite_entries(m: MatrixView) -> bool {
    m.iter().all(|x| x.is_finite())
}

pub fn subdiag_convergence(m: MatrixView, eps: f64) -> bool {
    let n = m.shape()[0];
    for i in 0..n - 1 {
        if m[[i + 1, i]] >= eps {
            return false;
        }
    }
    return true;
}
