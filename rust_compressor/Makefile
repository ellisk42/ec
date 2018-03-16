default: rust_compressor

rust_compressor: src/main.rs
	cargo build --release
	mv target/release/rust_compressor ./
	strip rust_compressor
