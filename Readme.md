# Bunch of general information

To do.

## Geometry branch specific information

Currently in order to build the solver on a fresh opam switch I needed the following packages (anecdotal data from Arch x64, assuming you have `opam`):

```bash
opam update                 # Seriously, do that one
opam switch 4.06.1+flambda  # caml.inria.fr/pub/docs/manual-ocaml/flambda.html
eval `opam config env`      # *sight*
opam install ppx_jane core re2 yojson vg cairo2
```

Now try to run `make` in the root folder, it *should* print many lines, some warnings but no error, and you *should* get a binary "solver" in the current folder.
