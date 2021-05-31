#[cfg(test)]
mod tests {
    use crate::types::*;
    use crate::algorithms::*;
    use crate::checks::*;

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
                   [13., 14., 15., 16.]];
        let mut b = a.clone();
        let mut c = Matrix::eye(4);
        reduce_to_hessenberg_form(b.view_mut(), c.view_mut());

        assert!(diff_subtriag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    fn test_qr_hess() {
        let a = ndarray::array![
                   [3., 7., 5.],
                   [-1., -3., -4.],
                   [0., 1., -9.]];
        let mut b = a.clone();
        let mut c = Matrix::eye(3);
        qr_algorithm_hessenberg(b.view_mut(), c.view_mut(), &OPTS);

        assert!(diff_triag(b.view()) < EPS);
        assert!(diff_unit(c.view()) < EPS);
        assert!(diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()) < EPS);
    }

    #[test]
    #[should_panic]
    fn qr_francis_test() {
        let a = ndarray::array![
            [0.162498, 0.331369, 0.0676646, 0.0105083, 0.735193, 0.722358],
            [0.0561499, 0.433707, 0.893129, 0.0968874, 0.0826798, 0.176299],
            [0.440105, 0.400527, 0.819697, 0.808964, 0.0544861, 0.037374],
            [0.965566, 0.563224, 0.28463, 0.169597, 0.936471, 0.41779],
            [0.325168, 0.27005, 0.234422, 0.0365608, 0.830469, 0.549708],
            [0.0156655, 0.190786, 0.317706, 0.166321, 0.704507, 0.36511]] * 20.;
        let mut b = a.clone();
        let mut c = Matrix::eye(6);
        reduce_to_hessenberg_form(b.view_mut(), c.view_mut());
        qr_algorithm_francis(b.view_mut(), c.view_mut(), &OPTS);
        // francis_block_reduction(b.view_mut(), c.view_mut(), EPS);

        println!("{:.2}", b);
        // println!("{:}", diff_rel(a.view(), c.dot(&b).dot(&c.t()).view()));

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
                    [0.121387 , 0.232579 , 0.0592648 , 0.165147],
                    [0.681462 , 0.255513 , 0.784051 , 0.867483],
                    [0.430509 , 0.355751 , 0.459837 , 0.938232],
                    [0.0703156 , 0.229938 , 0.441838 , 0.961529],
                    [0.896292 , 0.0657661 , 0.356057 , 0.254651],
                    [0.0963825 , 0.443201 , 0.793147 , 0.720426]
                ]).t().into_owned();

        let (u, s, vt) = svd(a.view());
        let mut ss = crate::Matrix::zeros((4, 6));
        for i in 0..4 {
            ss[[i, i]] = s[i];
        }

        assert!(diff_rel(a.view(), u.dot(&ss).dot(&vt).view()) < EPS);
        assert!(diff_unit(u.view()) < EPS);
        assert!(diff_unit(vt.view()) < EPS);
    }

    /*    #[test]
        fn uwu() {
            let mut a = ndarray::array![[0.445859, 0.0945522], [-1.396333, 0.22969]];
            let mut b = ndarray::array![[1., 0.], [0., 1.]];
            francis_block_reduction(a.view_mut(), b.view_mut(), EPS);
        }*/
}
