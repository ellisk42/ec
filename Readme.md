# Bunch of general information

To do.

### Build solver

Currently in order to build the solver on a fresh opam switch I needed the
following packages (anecdotal data from Arch x64, assuming you have `opam`):

```bash
opam update                 # Seriously, do that one
opam switch 4.06.1+flambda  # caml.inria.fr/pub/docs/manual-ocaml/flambda.html
eval `opam config env`      # *sight*
opam install ppx_jane core re2 yojson vg cairo2 camlimages menhir
```

Now try to run `make` in the root folder, it should build several ocaml
binaries.

### Build rust compressor

Get Rust (e.g. `curl https://sh.rustup.rs -sSf | sh` according to
[https://www.rust-lang.org/](https://www.rust-lang.org/en-US/install.html))

Now running make in the `rust_compressor` folder should install the right
packages and build the binary.

