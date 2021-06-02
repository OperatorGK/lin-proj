use crate::implementation::checks::finite_entries;
use crate::*;

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

#[inline]
pub fn hessenberg_form_inplace(m: MatrixViewMut) -> Result<Matrix> {
    hessenberg_form_inplace_opts(m, &DEFAULT_OPTS)
}

#[inline]
pub fn hessenberg_form_opts(m: MatrixView, opts: &QROptions) -> Result<(Matrix, Matrix)> {
    let mut t = m.into_owned();
    let u = hessenberg_form_inplace_opts(t.view_mut(), opts)?;
    Ok((t, u))
}

#[inline]
pub fn hessenberg_form(m: MatrixView) -> Result<(Matrix, Matrix)> {
    hessenberg_form_opts(m, &DEFAULT_OPTS)
}
