all:
	cd solvers && \
	  jbuilder build solver.exe && \
	  jbuilder build geomDrawLambdaString.exe && \
	  jbuilder build geomDrawFile.exe && \
	  cp _build/default/solver.exe ../solver && \
	  cp _build/default/geomDrawLambdaString.exe \
	    ../geomDrawLambdaString && \
	  cp _build/default/geomDrawFile.exe \
	    ./behaviouralData/geomDrawFile && \
	  strip ../solver && \
	  strip ../geomDrawLambdaString && \
	  strip ./behaviouralData/geomDrawFile

drawFile:
	cd solvers && \
	  jbuilder build geomDrawFile.exe && \
	  cp _build/default/geomDrawFile.exe \
	    ./behaviouralData/geomDrawFile

clean:
	cd solvers && jbuilder clean
	rm -f geomDrawLambdaString
	rm -f solver
