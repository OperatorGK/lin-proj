#[cfg(test)]
mod tests {
    use crate::algorithms::*;
    use crate::checks::*;
    use crate::types::*;

    const OPTS: QROptions = crate::algorithms::DEFAULT_OPTS;
    const EPS: f64 = OPTS.eps;

    #[test]
    fn test_gram_schmidt() {
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let (q, r) = qr_decompose_gram_schmidt(a.view());

        assert!(diff_unit(q.view()) < EPS);
        assert!(diff_triag(r.view()) < EPS);
        assert!(diff_rel(a.view(), q.dot(&r).view()) < EPS);
    }

    #[test]
    fn test_qr_naive() {
        let a = ndarray::array![[2., -3.], [-1., 1.]];
        let mut b = a.clone();
        let mut c = Matrix::eye(2);
        qr_algorithm_naive(b.view_mut(), c.view_mut(), &OPTS);

        assert!(diff_triag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    fn test_hess_form() {
        let a = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]
        ];
        let mut b = a.clone();
        let mut c = Matrix::eye(4);
        reduce_to_hessenberg_form(b.view_mut(), c.view_mut());

        assert!(diff_subtriag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_hess() {
        let a = ndarray::array![[3., 7., 5.], [-1., -3., -4.], [0., 1., -9.]];
        let mut b = a.clone();
        let mut c = Matrix::eye(3);
        qr_algorithm_hessenberg(b.view_mut(), c.view_mut(), &OPTS);

        assert!(diff_triag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_francis() {
        let a = ndarray::array![
            [7., 3., 4., -11., -9., -2.],
            [-6., 4., -5., 7., 1., 12.],
            [-1., -9., 2., 2., 9., 1.],
            [-8., 0., -1., 5., 0., 8.],
            [-4., 3., -5., 7., 2., 10.],
            [6., 1., 4., -11., -7., -1.]
        ];
        let mut b = a.clone();
        let mut c = Matrix::eye(6);
        reduce_to_hessenberg_form(b.view_mut(), c.view_mut());
        qr_algorithm_francis(b.view_mut(), c.view_mut(), &OPTS);
        francis_block_reduction(b.view_mut(), c.view_mut(), EPS);
        println!("{:.2}", b);

        assert!(diff_subtriag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_symmetric() {
        let a = ndarray::array![
            [1., 1., 1., 1.],
            [1., 2., 3., 4.],
            [1., 3., 6., 10.],
            [1., 4., 10., 20.]
        ];
        let mut b = a.clone();
        let mut c = Matrix::eye(4);
        reduce_to_hessenberg_form(b.view_mut(), c.view_mut());
        symmetric_qr_algorithm(b.view_mut(), c.view_mut(), &OPTS);

        assert!(diff_triag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    fn test_svd() {
        let a = (ndarray::array![
            [0.121387, 0.232579, 0.0592648, 0.165147],
            [0.681462, 0.255513, 0.784051, 0.867483],
            [0.430509, 0.355751, 0.459837, 0.938232],
            [0.0703156, 0.229938, 0.441838, 0.961529],
            [0.896292, 0.0657661, 0.356057, 0.254651],
            [0.0963825, 0.443201, 0.793147, 0.720426]
        ])
        .t()
        .into_owned();

        let (u, s, vt) = svd(a.view());
        let mut ss = crate::Matrix::zeros((4, 6));
        for i in 0..4 {
            ss[[i, i]] = s[i];
        }

        assert!(diff_rel(a.view(), u.dot(&ss).dot(&vt).view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_unit(vt.view()) < EPS);
    }
}
