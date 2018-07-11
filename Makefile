all:
	rm -f data/geom/logoDrawString
	cd solvers && \
	  jbuilder build solver.exe && \
	  jbuilder build geomDrawLambdaString.exe && \
	  jbuilder build geomDrawFile.exe && \
	  jbuilder build logoDrawString.exe && \
	  cp _build/default/solver.exe ../solver && \
	  cp _build/default/logoDrawString.exe \
	    ../logoDrawString && \
	  ln -s ../../logoDrawString \
	    ../data/geom/logoDrawString && \
	  cp _build/default/geomDrawLambdaString.exe \
	    ../geomDrawLambdaString && \
	  cp _build/default/geomDrawFile.exe \
	    ../data/geom/geomDrawFile

drawFile:
	cd solvers && \
	  jbuilder build geomDrawFile.exe && \
	  cp _build/default/geomDrawFile.exe \
	    ./behaviouralData/geomDrawFile
pb:
	cd towers && \
	  protoc --python_out=. cache.proto && \
	  ocaml-protoc -binary -ml_out ../solvers cache.proto
clean:
	cd solvers && jbuilder clean
	rm -f geomDrawLambdaString
	rm -f solver
	rm -f logoDrawString
	rm -f data/geom/geomDrawFile
	rm -f data/geom/logoDrawString
