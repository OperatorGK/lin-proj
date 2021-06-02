/// The type of floating-point vectors, re-exported from ndarray
pub type Vector = ndarray::Array1<f64>;

/// The type of immutable references to vectors
pub type VectorView<'a> = ndarray::ArrayView1<'a, f64>;

/// The type of mutable references to vectors
pub type VectorViewMut<'a> = ndarray::ArrayViewMut1<'a, f64>;

/// The type of floating-point matrices, re-exported from ndarray
pub type Matrix = ndarray::Array2<f64>;

/// The type of immutable references to matrices
pub type MatrixView<'a> = ndarray::ArrayView2<'a, f64>;

/// The type of mutable references to matrices
pub type MatrixViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;

/// The type of complex numbers, re-exported from num
pub type Complex = num::Complex<f64>;

/// Possible variants of QR algorithm
///
/// - `Naive` --- naive QR algorithm; O(n^3) operations per step, arbitrarily slow rate of convergence, often diverges.
/// - `Hessenberg` --- QR algorithm on Hessenberg matrices; O(n^3) reduction + O(n^2) operations per step, but same issues as the naive algorithm.
/// - `Francis` --- implicit double-shift QR algorithm; O(n^3) reduction + O(n^2) operations per step, usually linear covergence rate, almost always converges (but there are known counterexamples).
/// - `Symmetric` --- implicit shifit QR algorithm for symmetric matrices; O(n^3 reduction) + O(n^2) operations per step, usually linear covergence rate, always converges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QRAlgorithm {
    Naive,
    Hessenberg,
    Francis,
    Symmetric,
}

/// Algorithm options
///
/// - `eps` --- floating point epsilon, matrix entries with absolute value below it are considered zero.
/// - `iterations` --- (maximum) QR iterations performed. Francis and symmetric algorithms usually converge earlier.
/// - `algorithm` --- algorithm variant used.
/// - `do_safety_checks` --- whether to perform input validation and convergence checks.
/// - `zero_entries` --- whether to zero subepsilon entries.
/// - `accumulate_sim_transforms` --- whether to accumulate similarity transformations; returns an identity matrix in their place otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct QROptions {
    pub eps: f64,
    pub iterations: usize,
    pub algorithm: QRAlgorithm,
    pub do_safety_checks: bool,
    pub zero_entries: bool,
    pub accumulate_sim_transforms: bool,
}

/// Default algorithm options
pub const DEFAULT_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 100000,
    algorithm: QRAlgorithm::Francis,
    do_safety_checks: true,
    zero_entries: true,
    accumulate_sim_transforms: true,
};

/// Options for symmetric matrices and SVD decomposition
pub const SYMMETRIC_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 100000,
    algorithm: QRAlgorithm::Symmetric,
    do_safety_checks: true,
    zero_entries: true,
    accumulate_sim_transforms: true,
};

/// Options for eigenvalue calculation
pub const EIGENVALUE_OPTS: QROptions = QROptions {
    eps: 1e-8,
    iterations: 100000,
    algorithm: QRAlgorithm::Francis,
    do_safety_checks: true,
    zero_entries: false,
    accumulate_sim_transforms: false,
};
