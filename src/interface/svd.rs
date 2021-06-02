use crate::implementation::checks::finite_entries;
use crate::*;

/// Computes the SVD decomposition of a matrix, accepts options
///
/// Outputs `(U, S, V^T)` where `S` is a vector of singular values.
///
/// The SVD decomposition of matrix is a decomposition `A = U S V^T` where `U` and `V` are orthogonal and `S` has only diagonal entries.
/// Entries of `S` are called singular values of `A`, are non-negative and are sorted in the descending order.
///
/// Accepts any matrix with finite entries.
/// Performs O(n^3) operations.
pub fn svd_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Vector, Matrix)> {
    if opts.do_safety_checks && !finite_entries(m.view()) {
        return Err(QRError::NotFinite);
    }

    Ok(crate::implementation::svd::svd(m, opts))
}

/// Computes the SVD decomposition of a matrix
///
/// Outputs `(U, S, V^T)`.
/// Uses the default symmetric options.
/// See `svd_opts`.
pub fn svd(m: MatrixView) -> Result<(Matrix, Vector, Matrix)> {
    svd_opts(m, &SYMMETRIC_OPTS)
}
