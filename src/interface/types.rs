use std::fmt;

pub type Vector = ndarray::Array1<f64>;
pub type VectorView<'a> = ndarray::ArrayView1<'a, f64>;
pub type VectorViewMut<'a> = ndarray::ArrayViewMut1<'a, f64>;

pub type Matrix = ndarray::Array2<f64>;
pub type MatrixView<'a> = ndarray::ArrayView2<'a, f64>;
pub type MatrixViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;

pub type Complex = num::Complex<f64>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QRAlgorithm {
    Default,
    Naive,
    Hessenberg,
    Francis,
    Symmetric,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QROptions {
    pub eps: f64,
    pub iterations: usize,
    pub algorithm: QRAlgorithm,
    pub skip_safety_checks: bool,
    pub zero_entries: bool,
    pub accumulate_sim_transforms: bool,
}
