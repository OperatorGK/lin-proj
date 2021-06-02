use crate::implementation::common::extract_eigenvalues;
use crate::*;

/// Computes the matrix eigenvalues, accepts options
///
/// Complex number `l` is an eigenvalue of `A` if there exists a non-zero complex vector `v` for which `Av = lv`.
///
/// Calculates the Schur real form of a matrix and extracts the eigenvalues from its diagonal and subdiagonal.
/// Default algorithm used is Francis algorithm.
pub fn eigenvalues_opts(m: MatrixView, opts: &QROptions) -> Result<Vec<Complex>> {
    let (t, _) = schur_form_opts(m, opts)?;
    Ok(extract_eigenvalues(t.view()))
}

/// Computes the matrix eigenvalues
///
/// Uses default eigenvalue options.
/// See `eigenvalues_opts`.
pub fn eigenvalues(m: MatrixView) -> Result<Vec<Complex>> {
    eigenvalues_opts(m, &EIGENVALUE_OPTS)
}
