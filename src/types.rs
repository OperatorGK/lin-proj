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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QRError {
    NotSquare,
    NotSymmetric,
    ConvergenceFailed,
}

impl fmt::Display for QRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QRError::NotSquare => write!(f, "supplied matrix is not square"),
            QRError::NotSymmetric => write!(f, "supplied matrix is not symmetric"),
            QRError::ConvergenceFailed => write!(f, "algorithm failed to converge"),
        }
    }
}

pub type Result<T> = std::result::Result<T, QRError>;
