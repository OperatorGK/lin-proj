#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn uwu() {
        println!("QR -- Gramm-Schmidt");
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let (q, r) = crate::qr::qr_gs(a.view());
        println!("{}", q);
        println!("{}", r);

        println!("QR -- unshifted");
        let mut b = ndarray::array![[2., -3.], [-1., 1.]];
        let eig = crate::qr::qr_unshifted(b.view_mut(), 40);
        println!("{}", eig);

        println!("QR -- Hessenberg");
        let mut c = ndarray::array![
            [3., 7., 5.],
            [-1., -3., -4.],
            [0., 1., -9.]];
        crate::qr::qr_hess(c.view_mut(), 40);
        println!("{}", c);

        println!("Householder reduction");
        let mut d = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]];
        crate::qr::hess_form(d.view_mut());
        println!("{}", d);

        println!("QR -- double-shift");
        let mut e = ndarray::array![
            [-1., 2., 6.],
            [7., 0., 1.],
            [-0., -3., 3.]];
        crate::qr::qr_francis_shift(e.view_mut(), 1000000);
        println!("{}", e);
    }
}
