### Running using Singularity

If you don't want to manually install all the of the software dependencies locally you can instead do everything inside of singularity container. To build the container, you can use the recipe `singularity` in the repository, and just do (tested using singularity version 2.5):

```sudo singularity build container.img singularity
./container.img # will give you a shell into the container environment
```

# Bunch of general information

To do.

### Build solver

Currently in order to build the solver on a fresh opam switch I needed the
following packages (anecdotal data from Arch x64, assuming you have `opam`):

```bash
opam update                 # Seriously, do that one
opam switch 4.06.1+flambda  # caml.inria.fr/pub/docs/manual-ocaml/flambda.html
eval `opam config env`      # *sight*
opam install ppx_jane core re2 yojson vg cairo2 camlimages menhir ocaml-protoc zmq
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
```
git submodule update --recursive --init
```
from within the main project directory. You might need a recent version of git; 2.7.4 worked.

### Miscellaneous python dependencies

This should install all of the Python packages that you need. Not all
of these are needed for any particular domain, but all of these are
required by at least one domain.

```
pip install dill
pip install sexpdata
pip install Box2D-kengz
pip install graphviz
pip install pygame
pip install pycairo
pip install cairocffi
pip install psutil
conda install protobuf
pip install pypng
conda install pyzmq
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
pypy3 -m pip install --user psutil
```

### Credit of (most of) the `protonet` code

The `protonet-networks` folder contains some modifications over a big chunk of
code from [this repository](https://github.com/jakesnell/prototypical-networks), here is the attribution information :

> Code for the NIPS 2017 paper [Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf)

If you use that part of the code, please cite their paper paper, and check out
what they did:

```bibtex
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

### LICENSE of `protonets-networks` folder

MIT License

Copyright (c) 2017 Jake Snell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
