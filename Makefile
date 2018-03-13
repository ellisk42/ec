all:
	cd solvers && jbuilder clean && jbuilder build --verbose solver.exe && cp _build/default/solver.exe ../solver
