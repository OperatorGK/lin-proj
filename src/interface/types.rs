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
    pub do_safety_checks: bool,
    pub zero_entries: bool,
    pub accumulate_sim_transforms: bool,
}

pub const DEFAULT_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 100000,
    algorithm: QRAlgorithm::Default,
    do_safety_checks: true,
    zero_entries: true,
    accumulate_sim_transforms: true,
};

pub const SYMMETRIC_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 100000,
    algorithm: QRAlgorithm::Symmetric,
    do_safety_checks: true,
    zero_entries: true,
    accumulate_sim_transforms: true,
};

pub const EIGENVALUE_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 100000,
    algorithm: QRAlgorithm::Francis,
    do_safety_checks: true,
    zero_entries: false,
    accumulate_sim_transforms: false,
};
