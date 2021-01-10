pub type Vector = ndarray::Array1<f64>;
pub type VectorView<'a> = ndarray::ArrayView1<'a, f64>;
pub type VectorViewMut<'a> = ndarray::ArrayViewMut1<'a, f64>;

pub type Matrix = ndarray::Array2<f64>;
pub type MatrixView<'a> = ndarray::ArrayView2<'a, f64>;
pub type MatrixViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;
