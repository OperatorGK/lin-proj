use crate::*;
use crate::implementation::common::extract_eigenvalues;

pub fn eigenvalues_opts(m: MatrixView, opts: &QROptions) -> Result<Vec<Complex>> {
    let (t, _) = schur_form_opts(m, opts)?;
    Ok(extract_eigenvalues(t.view()))
}

pub fn eigenvalues(m: MatrixView) -> Result<Vec<Complex>> {
    eigenvalues_opts(m, &EIGENVALUE_OPTS)
}
