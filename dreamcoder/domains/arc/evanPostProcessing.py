


frontiersOverTime = convertFrontiersOverTimeToJson(frontierOverTime)

grammarJson = topDownGrammar.jsonWithTypes()
print(grammarJson)
productions = []

toReturn = {}

for task in frontierOverTime.keys():
    taskFrontiers = frontierOverTime[task]
    manuallySolved = str(task.name) in list(manuallySolvedTasks.keys())

    humanSolution = None
    if manuallySolved:
        humanProgram = parseHandwritten(str(task.name))
        usesIdx = getProgramUses(humanProgram, arrow(tgridin, tgridout), topDownGrammar)
        humanSolution = (len(usesIdx), usesIdx)

    taskData = {"manuallySolved": manuallySolved, "humanSolution": humanSolution, "programsOverTime": []}
    toReturn[str(task.name)] = taskData

    for frontier in taskFrontiers:
        if len(frontier.entries) > 0:
            bestProgram = frontier.bestPosterior.program
            usesIdx = getProgramUses(bestProgram, arrow(tgridin, tgridout), topDownGrammar)
            taskData["programsOverTime"].append((len(usesIdx), usesIdx))

for task in toReturn.keys():
    if toReturn[task]["manuallySolved"]:
        pass
        # print(task, toReturn[task])

print(len(grammarJson['productions']))
print(len(topDownGrammar.primitives))

for i,p in enumerate([str(p) for p in topDownGrammar.primitives]):
    print(i, p)

with open(resumePath + resumeDirectory + 'grammarPrimitivesList.p', 'wb') as f:
    pickle.dump([str(p) for p in topDownGrammar.primitives], f)

with open(resumePath + resumeDirectory + 'discoveredPrograms.p', 'wb') as f:
    pickle.dump(toReturn, f)

json.dump(grammarJson, open('grammar.json', 'w'))