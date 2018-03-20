c = corebuild -pkg yojson -pkg re2 -tag unsafe -tag "optimize(3)"

type:
	$(c) type.native; cp type.native test
TikZ:
	$(c) TikZ.native; cp TikZ.native test
solver:
	$(c)  solver.native; cp solver.native ../solver
profile:
	$(c) solver.p.native; cp solver.p.native ../solver
utilities:
	$(c) utils.native; cp utils.native test
timeout:
	$(c) timeout.native; cp timeout.native test
differentiation:
	$(c) differentiation.native; cp differentiation.native test
sfg:
	$(c) sfg.native; cp sfg.native test
text:
	$(c) string_transformation.native; cp string_transformation.native test
combinator:
	$(c) combinator.native; cp combinator.native test
program:
	$(c) program.native; cp program.native test
enumeration:
	$(c) enumeration.native; cp enumeration.native test
compression:
	$(c) compression.native; cp compression.native test
task:
	$(c) task.native; cp task.native test
list:
	$(c) list_synthesizer.native; cp list_synthesizer.native test
polynomial:
	$(c) polynomial.native; cp polynomial.native test
clean:
	rm -rf _build test *.native solver
run:
	time ./test

runcli:
	ocamlbuild -use-menhir -tag thread -use-ocamlfind \
	    -package camlimages \
	    -package camlimages.png \
	    -package gg \
	    -package vg \
	    -package vg.cairo \
	    -package vg.svg   \
	    geomDrawLambdaString.native

draw:
	ocamlbuild -use-menhir -tag thread -use-ocamlfind \
	    -package camlimages \
	    -package camlimages.png \
	    -package gg \
	    -package vg \
	    -package vg.cairo \
	    -package vg.svg   \
	    geomDrawFile.native
	mv geomDrawFile.native ./behaviouralData/geomDrawFile

generate:
	ocamlbuild -use-menhir -tag thread -use-ocamlfind \
	    -package camlimages \
	    -package camlimages.png \
	    -package gg \
	    -package vg \
	    -package vg.cairo \
	    -package vg.svg   \
	    geomGenerate.native

test:
	ocamlbuild -use-menhir -tag thread -use-ocamlfind \
	    -package camlimages \
	    -package camlimages.png \
	    -package gg \
	    -package vg \
	    -package vg.cairo \
	    -package vg.svg   \
	    geomTest.native
