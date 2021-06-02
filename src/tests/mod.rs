#[cfg(test)]
mod tests {
    use crate::*;
    use crate::implementation::checks::*;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    const OPTS: QROptions = QROptions { iterations: 1000, ..DEFAULT_OPTS };
    const EPS: f64 = 1e-4;

    #[test]
    fn test_gram_schmidt() {
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let (q, r) = qr_decomposition(a.view()).unwrap();

        assert!(diff_unit(q.view()) < EPS);
        assert!(diff_triag(r.view()) < EPS);
        assert!(diff_rel(a.view(), q.dot(&r).view()) < EPS);
    }

    #[test]
    fn test_qr_naive() {
        let a = ndarray::array![[2., -3.], [-1., 1.]];
        let opts = QROptions { algorithm: QRAlgorithm::Naive, ..OPTS };
        let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

        assert!(diff_triag(t.view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
    }

    #[test]
    fn test_hess_form() {
        let a = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ];
        let (t, u) = hessenberg_form(a.view()).unwrap();

        assert!(diff_subtriag(t.view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_hess() {
        let a = ndarray::array![[3., 7., 5.], [-1., -3., -4.], [0., 1., -9.]];
        let opts = QROptions { algorithm: QRAlgorithm::Hessenberg, ..OPTS };
        let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

        assert!(diff_triag(t.view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_francis() {
        let a = Array::random([100, 100], Uniform::new(-10., 10.));
        let opts = QROptions { algorithm: QRAlgorithm::Francis, ..OPTS };
        let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

        assert!(diff_subtriag(t.view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_symmetric() {
        let a = ndarray::array![
            [1., 1., 1., 1.],
            [1., 2., 3., 4.],
            [1., 3., 6., 10.],
            [1., 4., 10., 20.]
        ];
        let opts = QROptions { algorithm: QRAlgorithm::Symmetric, ..OPTS };
        let (t, u) = schur_form_opts(a.view(), &opts).unwrap();

        assert!(diff_triag(t.view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_rel(a.view(), u.dot(&t).dot(&u.t()).view()) < EPS);
    }

    #[test]
    fn test_svd() {
        let a = ndarray::array![
            [0.121387, 0.232579, 0.0592648, 0.165147],
            [0.681462, 0.255513, 0.784051, 0.867483],
            [0.430509, 0.355751, 0.459837, 0.938232],
            [0.0703156, 0.229938, 0.441838, 0.961529],
            [0.896292, 0.0657661, 0.356057, 0.254651],
            [0.0963825, 0.443201, 0.793147, 0.720426]
        ];

        let (u, s, vt) = svd(a.view()).unwrap();
        let mut ss = crate::Matrix::zeros((6, 4));
        for i in 0..4 {
            ss[[i, i]] = s[i];
        }

        assert!(diff_rel(a.view(), u.dot(&ss).dot(&vt).view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_unit(vt.view()) < EPS);
    }
}
