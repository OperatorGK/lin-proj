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
        let b = ndarray::array![[2., -3.], [-1., 1.]];
        let eig = crate::qr::qr_unshifted(b.view(), 40);
        println!("{}", eig);

        println!("QR -- Hessenberg");
        let c = ndarray::array![
            [3., 7., 5.],
            [-1., -3., -4.],
            [0., 1., -9.]];
        let eig = crate::qr::qr_hess(c.view(), 40);
        println!("{}", eig);

        println!("Householder reduction");
        let mut d = ndarray::array![
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.]];
        crate::qr::hess_form(d.view_mut());
        println!("{}", d);
        println!("{}", crate::qr::qr_hess(d.view(), 40));
    }
}
