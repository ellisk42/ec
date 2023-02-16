.PHONY: all solvers setup-ocaml

all:
	cd solvers && \
	  eval `opam env --switch .` && \
	  dune build @install --profile=release && \
	  dune install --sections bin --bindir=`pwd`/../
	ln -fs ../../logoDrawString data/geom/logoDrawString

clean:
	rm -f clevrSolver clevrTest compression helmholtz logoDrawString \
	  protonet_tester re2Test solver test_clevr_primitives versionDemo \
	  data/geom/logoDrawString
	cd solvers && eval `opam env --switch .` && dune clean

setup-ocaml:
	opam update
	cd solvers && opam switch create --deps-only --locked --yes .
