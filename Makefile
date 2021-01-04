all:
	rm -f data/geom/logoDrawString
	cd solvers && \
	  jbuilder build solver.exe && \
		jbuilder build clevrSolver.exe && \
	  jbuilder build versionDemo.exe && \
	  jbuilder build helmholtz.exe && \
	  jbuilder build logoDrawString.exe && \
	  jbuilder build protonet-tester.exe && \
	  jbuilder build compression.exe && \
	  cp _build/default/compression.exe ../compression && \
	  cp _build/default/versionDemo.exe ../versionDemo && \
	  cp _build/default/solver.exe ../solver && \
		cp _build/default/clevrSolver.exe ../clevrSolver && \
	  cp _build/default/helmholtz.exe ../helmholtz && \
	  cp _build/default/protonet-tester.exe ../protonet-tester && \
	  cp _build/default/logoDrawString.exe \
	    ../logoDrawString && \
	  ln -s ../../logoDrawString \
	    ../data/geom/logoDrawString
			
copy:
				cp ../ec_language/ec_language/solver .
				cp ../ec_language/ec_language/compression .
				cp ../ec_language/ec_language/helmholtz .
				cp ../ec_language/ec_language/logoDrawString .
				cp ../ec_language/ec_language/data/geom/logoDrawString data/geom/logoDrawString
clean:
	cd solvers && jbuilder clean
	rm -f solver
	rm -f clevrSolver
	rm -f compression
	rm -f helmholtz
	rm -f logoDrawString
	rm -f data/geom/logoDrawString

compressionClean:
	cd solvers && jbuilder clean
	rm -f compression

compression:
	cd solvers && \
	jbuilder build compression.exe && \
	cp _build/default/compression.exe ../compression

lcTestClean:
	cd solvers && jbuilder clean
	rm -f lcTest

lcTest:
	cd solvers && \
	jbuilder build lcTest.exe && \
	cp _build/default/lcTest.exe ../lcTest

clevrTestClean: 
	cd solvers && jbuilder clean
	rm -f clevrTest

clevrTest:
	cd solvers && \
	jbuilder build clevrTest.exe && \
	cp _build/default/clevrTest.exe ../clevrTest

solverClean: 
	cd solvers && jbuilder clean
	rm -f solver

solver: 
	cd solvers && \
	jbuilder build solver.exe && \
	cp _build/default/solver.exe ../solver

clevrClean: 
	cd solvers && jbuilder clean
	rm -f clevrSolver

clevrSolver: 
	cd solvers && \
	jbuilder build clevrSolver.exe && \
	cp _build/default/clevrSolver.exe ../clevrSolver

clevrPrimitivesTest:
	cd solvers && jbuilder clean 
	rm -f test_clevr_primitives 
	cd solvers && jbuilder build test_clevr_primitives.exe && \
	cp _build/default/test_clevr_primitives.exe ../test_clevr_primitives

re2TestClean: 
	cd solvers && jbuilder clean
	rm -f re2Test
	
re2Test: 
		cd solvers && \
		jbuilder build re2Test.exe && \
		cp _build/default/re2Test.exe ../re2Test

re2PrimsClean: 
	cd solvers && jbuilder clean
	rm -f re2Primitives
	
re2Prims: 
		cd solvers && \
		jbuilder build re2Primitives.exe && \
		cp _build/default/re2Primitives.exe ../re2Primitives