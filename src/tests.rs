#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn uwu() {
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let (q, r) = crate::qr::qr_gs(a.view());
        println!("{}", q);
        println!("{}", r);

        let b = ndarray::array![[2., -3.], [-1., 1.]];
        let eig = crate::qr::qr_unshifted(b.view(), 4);
        println!("{}", eig);
    }
}
