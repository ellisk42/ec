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

### Installing submodules

Run:
```git submodule update --recursive --init
```
from within the main project directory. You might need a recent version of git; 2.7.4 worked.

### Blocks world domain

Implemented in `tower.py`. To install the dependencies, do:

```
pip install box2d
pip install pygame
pip install pycairo
pip install psutil
```

### PyPy

If for some reason you want to run something in pypy, install it from:
```
https://github.com/squeaky-pl/portable-pypy#portable-pypy-distribution-for-linux
```
Be sure to add `pypy3` to the path. Really though you should try to
use the rust compressor and the ocaml solver. You will have to
(annoyingly) install parallel libraries on the pypy side even if you
have them installed on the Python side:

```
pypy3 -m ensurepip
pypy3 -m pip install --user vmprof
pypy3 -m pip install --user dill
```
