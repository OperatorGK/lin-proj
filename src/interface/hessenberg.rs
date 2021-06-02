use crate::implementation::checks::finite_entries;
use crate::*;

/// Computes the Hessenberg form of a matrix in-place, accepts options
///
/// Outputs the similarity transformation matrix `U` if desired (`opts.accumulate_sim_transforms == true`).
///
/// The Hessenberg form is a matrix `H` which is almost triangular (no elements beneath the subdiagonal) and similar to the supplied matrix (`U H U^T = A`).
///
/// Accepts any square matrix with finite entries.
/// Performs O(n^3) operations.
#[inline]
pub fn hessenberg_form_inplace_opts(mut m: MatrixViewMut, opts: &QROptions) -> Result<Matrix> {
    if opts.do_safety_checks {
        if !finite_entries(m.view()) {
            return Err(QRError::NotFinite);
        }

        if !m.is_square() {
            return Err(QRError::NotSquare);
        }
    }

    let mut u = Matrix::eye(m.shape()[0]);
    crate::implementation::hessenberg::hessenberg_form(m.view_mut(), u.view_mut(), opts);
    if opts.zero_entries {
        crate::implementation::common::zero_subeps_entries(m.view_mut(), opts.eps);
    }

    Ok(u)
}

/// Computes the Hessenberg form of a matrix in-place
///
/// Outputs the similarity transformation matrix `U`.
/// Uses default options.
/// See `hessenberg_form_inplace_opts`.
#[inline]
pub fn hessenberg_form_inplace(m: MatrixViewMut) -> Result<Matrix> {
    hessenberg_form_inplace_opts(m, &DEFAULT_OPTS)
}

/// Computes the Hessenberg form of a matrix, accepts options
///
/// Outputs `(H, U)` where `H` is a Hessenberg matrix and `U` is orthogonal.
/// See `hessenberg_form_inplace_opts`.
#[inline]
pub fn hessenberg_form_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Matrix)> {
    let mut t = m.into_owned();
    let u = hessenberg_form_inplace_opts(t.view_mut(), opts)?;
    Ok((t, u))
}

/// Computes the Hessenberg form of a matrix
///
/// Outputs `(H, U)` where `H` is a Hessenberg matrix and `U` is orthogonal.
/// Uses default options.
/// See `hessenberg_form_inplace_opts`.
#[inline]
pub fn hessenberg_form(m: MatrixView) -> Result<(Matrix, Matrix)> {
    hessenberg_form_opts(m, &DEFAULT_OPTS)
}
