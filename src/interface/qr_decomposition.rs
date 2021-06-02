use crate::implementation::checks::finite_entries;
use crate::implementation::common::zero_subeps_entries;
use crate::*;

/// Computes the QR decomposition of a matrix, accepts options
///
/// Outputs (`Q`, `R`).
///
/// Decomposition `A = QR`, where `Q` is orthogonal and `R` is triagonal is called a QR decomposition.
///
/// Accepts any square matrix with finite entries.
/// Performs O(n^3) operations.
pub fn qr_decomposition_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Matrix)> {
    if opts.do_safety_checks {
        if !finite_entries(m) {
            return Err(QRError::NotFinite);
        }

        if !m.is_square() {
            return Err(QRError::NotSquare);
        }
    }

    let (mut q, r) = crate::implementation::qr_basic::qr_decomposition(m);
    if opts.zero_entries {
        zero_subeps_entries(q.view_mut(), opts.eps);
    }

    Ok((q, r))
}

/// Computes the QR decomposition of a matrix
///
/// Outputs (`Q`, `R`).
/// Uses default options.
/// See `qr_decomposition_opts`.
pub fn qr_decomposition(m: MatrixView) -> Result<(Matrix, Matrix)> {
    qr_decomposition_opts(m, &DEFAULT_OPTS)
}
