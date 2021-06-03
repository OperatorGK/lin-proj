# lin-proj

Сборка:

После [установки компилятора Rust](https://www.rust-lang.org/tools/install) необходимо переключить версию toolchain-а на nightly
командой `rustup default nightly`

После переключения проект собирается командой `cargo build --release`

Тестирование: `cargo test`

Бенчмарки: `cargo bench`

Документация: `cargo doc`, вывод в `/target/doc/lin_proj/*`, начальная страница `index.html`.
