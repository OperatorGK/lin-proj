use std::fmt;

/// Possible algorithm errors
///
/// - NotFinite --- supplied matrix has infinite or NaN entries.
/// - NotSquare --- supplied matrix is not square.
/// - NotSymmetric --- symmetric algorithm used, but the matrix is not symmetric.
/// - ConvergenceFailed --- algorithm failed to converge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QRError {
    NotFinite,
    NotSquare,
    NotSymmetric,
    ConvergenceFailed,
}

impl fmt::Display for QRError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            QRError::NotFinite => write!(f, "supplied matrix has infinite or nan entries"),
            QRError::NotSquare => write!(f, "supplied matrix is not square"),
            QRError::NotSymmetric => write!(f, "supplied matrix is not symmetric"),
            QRError::ConvergenceFailed => write!(f, "algorithm failed to converge"),
        }
    }
}

/// Result<T, QRError>
pub type Result<T> = std::result::Result<T, QRError>;
