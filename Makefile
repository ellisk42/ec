all:
	rm -f data/geom/logoDrawString
	cd solvers && \
	  jbuilder build solver.exe && \
	  jbuilder build versionDemo.exe && \
	  jbuilder build helmholtz.exe && \
	  jbuilder build logoDrawString.exe && \
	  jbuilder build protonet-tester.exe && \
	  jbuilder build compression.exe && \
	  cp _build/default/compression.exe ../compression && \
	  cp _build/default/versionDemo.exe ../versionDemo && \
	  cp _build/default/solver.exe ../solver && \
	  cp _build/default/helmholtz.exe ../helmholtz && \
	  cp _build/default/protonet-tester.exe ../protonet-tester && \
	  cp _build/default/logoDrawString.exe \
	    ../logoDrawString && \
	  ln -s ../../logoDrawString \
	    ../data/geom/logoDrawString

clean:
	cd solvers && jbuilder clean
	rm -f solver
	rm -f compression
	rm -f helmholtz
	rm -f logoDrawString
	rm -f data/geom/logoDrawString
