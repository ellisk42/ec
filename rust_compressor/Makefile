default: rust_compressor

rust_compressor: Cargo.toml Cargo.lock src/*.rs
	cargo build --release
	mv target/release/rust_compressor ./
	strip rust_compressor
