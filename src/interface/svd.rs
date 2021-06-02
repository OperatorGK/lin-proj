use crate::*;
use crate::implementation::checks::finite_entries;

pub fn svd_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Vector, Matrix)> {
    if opts.do_safety_checks && !finite_entries(m.view()) {
        return Err(QRError::NotFinite);
    }

    Ok(crate::implementation::svd::svd(m, opts))
}

pub fn svd(m: MatrixView) -> Result<(Matrix, Vector, Matrix)> {
    svd_opts(m, &SYMMETRIC_OPTS)
}
