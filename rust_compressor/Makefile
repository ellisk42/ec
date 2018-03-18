default: lock rust_compressor

lock:
	cargo update

rust_compressor: Cargo.lock src/main.rs
	cargo build --release
	mv target/release/rust_compressor ./
	strip rust_compressor
