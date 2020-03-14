yes = True

candidateSketches = [PQ() for _ in tasks]


# find and rank sketches
for sketch in g.getSketches():

	for n, task in enumerate(tasks):
		val = self.computeValue(sketch, task)
		candidateSketches[n].push(val, sketch )

		if len(candidateSketches[n]) > maxSketches[n]:
			candidateSketches[n].pssopMaximum()


for n, task in enumerate(tasks):
	
	# get best sketch?
	# enumerate from best sketch ??


	for sketch in candidateSketches:

		for prog in g.sketchEnumeration(self,context,environment,request,sketch,upperBound,
                           maximumDepth=20,
                           lowerBound=0.):
		




    frontiers = {tasks[n]: Frontier([e for _, e in hits[n]],
                                    task=tasks[n])
                 for n in range(len(tasks))}