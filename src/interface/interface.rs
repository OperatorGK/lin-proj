use crate::algorithms::*;
use crate::aux::extract_eigenvalues;
use crate::types::*;

pub mod types;
mod checks;

pub const DEFAULT_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 1000,
    algorithm: QRAlgorithm::Default,
    skip_safety_checks: false,
};

pub fn schur_form_inplace_unsafe_opts(mut m: MatrixViewMut, opts: &QROptions) -> Matrix {
    let mut u = Matrix::eye(m.shape()[0]);
    match opts.algorithm {
        QRAlgorithm::Default => {
            reduce_to_hessenberg_form(m.view_mut(), u.view_mut());
            qr_algorithm_francis(m.view_mut(), u.view_mut(), opts);
        }
        QRAlgorithm::Naive => {
            qr_algorithm_naive(m.view_mut(), u.view_mut(), opts);
        }
        QRAlgorithm::Hessenberg => {
            reduce_to_hessenberg_form(m.view_mut(), u.view_mut());
            qr_algorithm_hessenberg(m.view_mut(), u.view_mut(), opts);
        }
        QRAlgorithm::Francis => {
            reduce_to_hessenberg_form(m.view_mut(), u.view_mut());
            qr_algorithm_francis(m.view_mut(), u.view_mut(), opts);
            francis_block_reduction(m.view_mut(), u.view_mut(), opts.eps);
        }
        QRAlgorithm::Symmetric => {
            reduce_to_hessenberg_form(m.view_mut(), u.view_mut());
            symmetric_qr_algorithm(m.view_mut(), u.view_mut(), opts);
        }
    };
    u
}

pub fn schur_form_inplace_unsafe(m: MatrixViewMut) -> Matrix {
    schur_form_inplace_unsafe_opts(m, &DEFAULT_OPTS)
}

pub fn schur_form_inplace_opts(m: MatrixViewMut, opts: &QROptions) -> Result<Matrix> {
    if !m.is_square() {
        Result::Err(QRError::NotSquare)?;
    }
    if opts.algorithm == QRAlgorithm::Symmetric && diff_symm(m.view()) >= opts.eps {
        Result::Err(QRError::NotSymmetric)?;
    }

    let u = schur_form_inplace_unsafe_opts(m, opts);
    Result::Ok(u)
}

pub fn schur_form_inplace(m: MatrixViewMut) -> Result<Matrix> {
    schur_form_inplace_opts(m, &DEFAULT_OPTS)
}

pub fn schur_form_unsafe_opts(m: MatrixView, opts: &QROptions) -> (Matrix, Matrix) {
    let mut t = m.into_owned();
    let u = schur_form_inplace_unsafe_opts(t.view_mut(), opts);
    (t, u)
}

pub fn schur_form_unsafe(m: MatrixView) -> (Matrix, Matrix) {
    schur_form_unsafe_opts(m, &DEFAULT_OPTS)
}

pub fn schur_form_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Matrix)> {
    let mut t = m.into_owned();
    let u = schur_form_inplace_opts(t.view_mut(), opts)?;
    Result::Ok((t, u))
}

pub fn schur_form(m: MatrixView) -> Result<(Matrix, Matrix)> {
    schur_form_opts(m, &DEFAULT_OPTS)
}

pub fn svd(m: MatrixView) -> (Matrix, Vector, Matrix) {
    if m.shape()[0] > m.shape()[1] {
        let (ut, s, v) = crate::algorithms::svd(m.t());
        (ut.t().into_owned(), s, v.t().into_owned())
    } else {
        crate::algorithms::svd(m)
    }
}

pub fn eigenvalues(m: MatrixView) -> Result<Vec<Complex>> {
    let (t, _) = schur_form(m)?;
    let eig = extract_eigenvalues(t.view());
    Ok(eig)
}
