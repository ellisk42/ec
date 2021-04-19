import copy
import dill
import json
import math
import torch

from dreamcoder.grammar import Grammar, ContextualGrammar
from dreamcoder.program import Program
from dreamcoder.type import *
    
# Uses `parameters` to construct the checkpoint path
def checkpointPath(iteration, extra=""):
    parameters["iterations"] = iteration
    kvs = [
        "{}={}".format(
            ECResult.abbreviate(k),
            parameters[k]) for k in sorted(
            parameters.keys())]
    return "{}_{}{}.pickle".format(outputPrefix, "_".join(kvs), extra)

def resume_from_path(resume):
    try:
        resume = int(resume)
        path = checkpointPath(resume)
    except ValueError:
        path = resume
    with open(path, "rb") as handle:
        result = dill.load(handle)
    resume = len(result.grammars) - 1
    print("Loaded checkpoint from", path)
    grammar = result.grammars[-1] if result.grammars else grammar
    return result, grammar, result.grammars[0]

def getLearnedProductions(result, verbose=False):
    currentProductions = set()
    learnedProductions = {}
    for i,grammar in enumerate(result.grammars):
        learnedProductions[i] = {}
        productions = set([str(p[2]) for p in grammar.productions])
        if len(list(currentProductions)) == 0:
            currentProductions = productions
        else:
            if verbose:
                print('----------------------------------------------------{}-----------------------------------------------------'.format(i))        
            newProductions = productions.difference(currentProductions)
            for j,production in enumerate(newProductions):
                learnedProductions[i][j] = production
                if verbose:
                    print(j, production)
        currentProductions = currentProductions.union(productions)
    return learnedProductions


def getTrainFrontier(resumePath, n, verbose=False):
    result, resumeGrammar, preConsolidationGrammar = resume_from_path(resumePath)
    firstFrontier = [frontiers[0] for (key,frontiers) in result.frontiersOverTime.items() if len(frontiers[0].entries) > 0]
    allFrontiers = [frontier for (task,frontier) in result.allFrontiers.items() if len(frontier.entries) > 0]
    # expandedFrontier = expandFrontier(firstFrontier, n)
    if verbose:
        print(result.learningCurve)
    learnedProductions = getLearnedProductions(result)
    return result, firstFrontier, allFrontiers, result.frontiersOverTime, resumeGrammar, preConsolidationGrammar, result.recognitionModel, learnedProductions

def getTask(specificTask, allTasks):
    for task in allTasks:
        if str(task) == str(specificTask):
            return task
    raise Exception

def scoreProgram(p, recognizer=None, grammar=None, taskName=None, task=None):
    if taskName:
        task = getTask(taskName, trainTasks)
    if recognizer is not None:
        grammar = recognizer.grammarOfTask(task).untorch()
    if grammar is None:
        return 0
    ll = grammar.logLikelihood(arrow(tgridin, tgridout), p)
    return ll

def normalizeProductions(grammar):
    z = lse([l for l,_,p in grammar.productions])
    normalizedProductions = [(p[0]-z, p[2]) for p in grammar.productions]
    return Grammar.fromProductions(normalizedProductions)

def upweightProduction(name, scaleFactor, grammar):
    productions = [(p[0],p[2]) for p in grammar.productions]
    for i in range(len(productions)):
        if str(productions[i][1]) == name:
            print('production before: {}'.format(productions[i][0]))
            productions[i] = (productions[i][0] + math.log(scaleFactor), productions[i][1])
            print('production after: {}'.format(productions[i][0]))
    return Grammar.fromProductions(productions)

def upweightConditionalProduction(parentPrimitive, argumentIndex, production, scaleFactor, contextualGrammar):
    primitiveGrammar = contextualGrammar.library[parentPrimitive][argumentIndex]
    print('primitiveGrammar before: ', primitiveGrammar)
    newPrimitiveGrammar = upweightProduction(production, scaleFactor, primitiveGrammar)
    print('conditionalGrammar after: ', newPrimitiveGrammar)
    newContextualGrammar = copy.deepcopy(contextualGrammar)
    newContextualGrammar.library[parentPrimitive][argumentIndex] = newPrimitiveGrammar
    return newContextualGrammar

def upweightConditionalProductions(parent2UpdateProduction, scaleFactor, contextualGrammar):
    currentGrammar = contextualGrammar
    for ((parentPrimitive, argIndex),production) in parent2UpdateProduction.items():
        currentGrammar = upweightConditionalProduction(parentPrimitive, argIndex, production, scaleFactor, currentGrammar)
    return currentGrammar

def evaluateGrammars(dreamcoderFrontier, manuallySolvedTasks, grammar1=None, grammar2=None, recognizer1=None, recognizer2=None):

    print('\n ------------------------------ Solved by Dreamcoder ------------------------------------ \n')
    averageLLBefore, averageLLAfter = 0,0
    for frontier in dreamcoderFrontier:
        llBefore = scoreProgram(frontier.entries[0].program, grammar=grammar1, recognizer=recognizer1)
        llAfter = scoreProgram(frontier.entries[0].program, grammar=grammar2, recognizer=recognizer2)
        if llAfter == 0:
            print("{}: {}".format(frontier.task, llBefore))
        else:
            print("{}: {} -> {}".format(frontier.task.name, llBefore, llAfter))
        print("Program: {}\n".format(frontier.entries[0].program))
        averageLLBefore += llBefore
        averageLLAfter += llAfter

    print("Solved {} tasks".format(len(dreamcoderFrontier)))
    if llAfter == 0:
        print('Average LL: {}'.format(averageLLBefore / len(dreamcoderFrontier)))
    else:
        print('Average LL: {} -> {}'.format(averageLLBefore / len(dreamcoderFrontier), (averageLLAfter / len(dreamcoderFrontier))))

    print('\n ------------------------------ Manually Solved ------------------------------------ \n')        


    solvedByDreamcoder = 0
    averageLLBefore, averageLLAfter = 0,0
    for task,program in manuallySolvedTasks.items():
        if task not in [frontier.task.name for frontier in dreamcoderFrontier]:
            p = Program.parse(manuallySolvedTasks[task])
            llBefore = scoreProgram(p, grammar=grammar1, recognizer=recognizer1)
            llAfter = scoreProgram(p, grammar=grammar2, recognizer=recognizer2)
            if llAfter == 0:
                print("{}: {}".format(task, llBefore))
            else:
                print("{}: {} -> {}".format(task, llBefore, llAfter))
            print("Program: {}\n".format(p))
            averageLLBefore += llBefore
            averageLLAfter += llAfter
        else:
            solvedByDreamcoder += 1
    
    print("{} of {} manually written solutions were found by Dreamcoder".format(solvedByDreamcoder, len(manuallySolvedTasks)))
    numUnsolved = len(manuallySolvedTasks) - solvedByDreamcoder
    if llAfter == 0:
        print('Average LL: {}'.format(averageLLBefore / numUnsolved))
    else:
        print('Average LL: {} -> {}'.format(averageLLBefore/numUnsolved, averageLLAfter/numUnsolved))

def convertFrontiersOverTimeToJson(frontiersOverTime):
    frontiersOverTimeJson = {}
    numFrontiers = len(list(frontiersOverTime.values())[0])
    # print('{} frontiers per task'.format(numFrontiers))
    for task,frontiers in frontiersOverTime.items():
        # print('frontiers: ', frontiers)
        frontiersOverTimeJson[task.name] = {i:str(frontier.entries[0].program) + '\n' + str(frontier.entries[0].logPosterior) for i,frontier in enumerate(frontiers) if len(frontier.entries) > 0}
    return frontiersOverTimeJson

def getProgramUses(program, request, grammar):
    ls = grammar.closedLikelihoodSummary(request, program)
    def uses(summary):
        if hasattr(summary, 'uses'): 
            return torch.tensor([ float(int(p in summary.uses))
                                  for p in grammar.primitives])
        assert hasattr(summary, 'noParent')
        u = uses(summary.noParent) + uses(summary.variableParent)
        for ss in summary.library.values():
            for s in ss:
                u += uses(s)
        return u
    u = uses(ls)
    # u[u > 1.] = 1.
    usesIdx = torch.nonzero(u).squeeze().tolist()
    if isinstance(usesIdx, int):
        usesIdx = [usesIdx]
    return usesIdx

def parseHandwritten(taskName):
    # if taskName in ["007bbfb7.json", "ea786f4a.json"]:
    #     return None
    # else:
    print(taskName)
    program = Program.parse(manuallySolvedTasks[taskName])
    return program

def createResultsJsonForInterface(resumePath, resumeDirectory, pickledFile):

        result, firstFrontier, allFrontiers, frontierOverTime, topDownGrammar, preConsolidationGrammar, resumeRecognizer, learnedProductions = getTrainFrontier(resumePath + resumeDirectory + pickledFile, 0)

        with open(resumePath + resumeDirectory + 'frontiersOverTime.json', 'w') as fp:
            json.dump(frontiersOverTime, fp)

        with open(resumePath + resumeDirectory + 'ecResults.json', 'w') as fp:
            json.dump({'learnedProductions':learnedProductions, 'frontiersOverTime':frontiersOverTime}, fp)

        return

def checkFrontiersOnEvalExamples(frontiers):
    # TODO: Fix to use ocaml implementation of program since not all python primitives
    # are implemented

    percentCorrect = 0.0
    for frontier in frontiers:
        print(type(frontier.task))
        task = getTask(frontier.task, trainTasks)
        print(type(task))
        program = frontier.topK(1).entries[0].program
        print('Task: {}'.format(task.name))
        print("Program: {}".format(program))

        isCorrect = task.checkEvalExamples(program, timeout=1.0)
        print("Correct?: {}".format(isCorrect))
        if isCorrect:
            percentCorrect += 1.0
    return percentCorrect / len(frontiers)

    # # Tasks I'm not solving
    # # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=preConsolidationGrammar.insideOutside(firstFrontier, 30, iterations=1))
    # # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=preConsolidationGrammar.insideOutside(firstFrontier, 1))

    # # How does contextual model do?
    # # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=topDownGrammar, recognizer2=resumeRecognizer)

    # parent2UpdateProduction = {
    #     # (Primitive('blocks_to_original_grid', arrow(tblocks, tbool, tbool, tgridout),  None), 0): 'map_tbs',
    #     # (Primitive("map_tbs", arrow(ttbs, arrow(tblock, tblock, tblock), tbool, tblocks), None),1): 'move_until_touches_block',

    #     (Primitive('blocks_to_original_grid', arrow(tblocks, tbool, tbool, tgridout),  None), 0): 'map_blocks',
    #     # (Primitive("map_blocks", arrow(tblocks, arrow(tblock, tblock), tblocks), None),1): 'fill_color',
    #     (Primitive('fill_color', arrow(tblock, tcolor, tblock), None), 1): 'blue',
    #     # (Primitive("filter_blocks", arrow(tblocks, arrow(tblock, tbool), tblocks), None),1): 'touches_any_boundary',

    #     # (Primitive('blocks_to_original_grid', arrow(tblocks, tbool, tbool, tgridout),  None), 0): 'map_blocks',
    #     # (Primitive("map_tiles", arrow(ttiles, arrow(ttile, tblock), tblocks), None),1): 'extend_towards_until',
    #     # (Primitive("make_colorpair", arrow(tcolor, tcolor, tcolorpair), None), 1): 'invisible',

    #     # (Primitive('fill_color', arrow(tblock, tcolor, tblock), None), 1): 'teal',
    #     # (Primitive("map_blocks", arrow(tblocks, arrow(tblock, tblock), tblocks), None),1): 'fill_snakewise',
    #     # (Primitive("filter_template_block", arrow(tblocks, arrow(tblock, tbool), ttbs), None),0):'find_blocks_by_inferred_b',
    #     # (Primitive("filter_template_block", arrow(tblocks, arrow(tblock, tbool), ttbs), None),1):'has_min_tiles',
    #     # (Primitive("filter", arrow(tblocks, arrow(tblock, tblock), tblocks), None),0): 'extend_towards_until'
    # }
    # preConsolidationGrammarInsideOut = preConsolidationGrammar.insideOutside(firstFrontier,1)
    # contextualGrammar = ContextualGrammar.fromGrammar(preConsolidationGrammarInsideOut)
    # newContextualGrammar = upweightConditionalProductions(parent2UpdateProduction, 100, contextualGrammar)
    # evaluateGrammars(firstFrontier, manuallySolvedTasks, grammar1=preConsolidationGrammarInsideOut, grammar2=newContextualGrammar)