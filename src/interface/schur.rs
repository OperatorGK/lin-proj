use crate::implementation::checks::{diff_subtriag, diff_symm, diff_triag, finite_entries};
use crate::implementation::common::zero_subeps_entries;
use crate::implementation::francis::{francis_block_reduction, qr_algorithm_francis};
use crate::implementation::hessenberg::{hessenberg_form, qr_algorithm_hessenberg};
use crate::implementation::qr_basic::qr_algorithm_naive;
use crate::implementation::qr_symmetric::qr_algorithm_symmetric;
use crate::*;

/// Computes the (real) Schur form of a matrix in-place, accepts options
///
/// Outputs the similarity transformation matrix `U` if desired (`opts.accumulate_sim_transforms == true`).
///
/// Real Schur form of a matrix is decomposition `A = U T U^T`, where `U` is orthogonal and `T` is (pseudo-)triangular.
/// Eigenvalues of `A` are located on the diagonal of `T`.
/// When using the Francis algorithm, `T` may have `2 by 2` blocks on its diagonal representing complex eigenvalues.
///
/// Accepts any square matrix with finite entries, but convergence is not guaranteed.
/// See `QRAlgorithm` description for algorithm details.
pub fn schur_form_inplace_opts(mut m: MatrixViewMut, opts: &QROptions) -> Result<Matrix> {
    if opts.do_safety_checks {
        if !finite_entries(m.view()) {
            return Err(QRError::NotFinite);
        }

        if !m.is_square() {
            return Err(QRError::NotSquare);
        }

        if opts.algorithm == QRAlgorithm::Symmetric && !(diff_symm(m.view()) < opts.eps.sqrt()) {
            return Err(QRError::NotSymmetric);
        }
    }

    let mut u = Matrix::eye(m.shape()[0]);
    match opts.algorithm {
        QRAlgorithm::Francis => {
            hessenberg_form(m.view_mut(), u.view_mut(), opts);
            qr_algorithm_francis(m.view_mut(), u.view_mut(), opts);
            francis_block_reduction(m.view_mut(), u.view_mut(), opts);
        }
        QRAlgorithm::Naive => {
            qr_algorithm_naive(m.view_mut(), u.view_mut(), opts);
        }
        QRAlgorithm::Hessenberg => {
            hessenberg_form(m.view_mut(), u.view_mut(), opts);
            qr_algorithm_hessenberg(m.view_mut(), u.view_mut(), opts);
        }
        QRAlgorithm::Symmetric => {
            hessenberg_form(m.view_mut(), u.view_mut(), opts);
            qr_algorithm_symmetric(m.view_mut(), u.view_mut(), opts);
        }
    }

    if opts.zero_entries {
        zero_subeps_entries(m.view_mut(), opts.eps);
    }

    if opts.do_safety_checks {
        let diff = match opts.algorithm {
            QRAlgorithm::Francis => diff_subtriag(m.view()),
            _ => diff_triag(m.view()),
        };

        if !(diff < opts.eps.sqrt()) {
            return Err(QRError::ConvergenceFailed);
        }
    }

    Ok(u)
}

/// Computes the (real) Schur form of a matrix in-place
///
/// Outputs `U`.
/// Uses default options.
/// See `schur_form_inplace_opts`.
pub fn schur_form_inplace(m: MatrixViewMut) -> Result<Matrix> {
    schur_form_inplace_opts(m, &DEFAULT_OPTS)
}

/// Computes the (real) Schur form of a matrix, accepts options
///
/// Outputs `(T, U)`.
/// See `schur_form_inplace_opts`.
pub fn schur_form_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Matrix)> {
    let mut t = m.into_owned();
    let u = schur_form_inplace_opts(t.view_mut(), opts)?;
    Ok((t, u))
}

/// Computes the (real) Schur form of a matrix
///
/// Outputs `(T, U)`.
/// Uses default options.
/// See `schur_form_inplace_opts`.
pub fn schur_form(m: MatrixView) -> Result<(Matrix, Matrix)> {
    schur_form_opts(m, &DEFAULT_OPTS)
}
