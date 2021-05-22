#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn uwu() {
        let opts = crate::qr::DEFAULT_OPTS;

        println!("QR -- Gramm-Schmidt");
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let (q, r) = crate::qr::qr_decompose_gram_schmidt(a.view());
        println!("{}", q);
        println!("{}", r);

        println!("QR -- unshifted");
        let mut b = ndarray::array![[2., -3.], [-1., 1.]];
        let eig = crate::qr::qr_algorithm_naive(b.view_mut(), &opts);
        println!("{}", eig);

        println!("QR -- Hessenberg");
        let mut c = ndarray::array![
            [3., 7., 5.],
            [-1., -3., -4.],
            [0., 1., -9.]];
        crate::qr::qr_algorithm_hessenberg(c.view_mut(), &opts);
        println!("{}", c);

        println!("Householder reduction");
        let mut d = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]];
        crate::qr::reduce_to_hessenberg_form(d.view_mut());
        println!("{}", d);

        println!("QR -- double-shift");
        let mut e = ndarray::array![
            [-1., 2., 6.],
            [7., 0., 1.],
            [-0., -3., 3.]];
        crate::qr::qr_algorithm_francis(e.view_mut(), &opts);
        println!("{}", e);
    }
}
