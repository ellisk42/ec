all:
	cd solvers && \
	  jbuilder clean && \
	  jbuilder build --verbose solver.exe && \
	  jbuilder build --verbose geomDrawLambdaString.exe && \
	  cp _build/default/solver.exe ../solver && \
	  cp _build/default/geomDrawLambdaString.exe \
	    ../geomDrawLambdaString

clean:
	cd solvers && jbuilder clean
	rm -f geomDrawLambdaString
	rm -f solver
