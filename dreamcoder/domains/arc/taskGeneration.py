def replaceColors(inputGrid, colorMap):
    """
    Replaces colors in the inputGrid by the corresponding value in colorMap (where each item is (oldColor, newColor)).
    """

    return inputGrid.fromPoints({key:colorMap.get(oldColor,oldColor) for (key,oldColor) in inputGrid.points.items()})

def getColorsToReplace(examples, doNotReplace=[0]):
    """ 
    Returns a set of the non-special colors in examples. The non-special colors are colors that are in both the input and output grids.
    """

    allColors = set()
    for example in examples:
        inputGrid = example[0][0]
        allColors = allColors.union(set(inputGrid.points.values()))
    return allColors - set(doNotReplace)

class LessPermutationsThanRequiredExamples(Exception):
    pass

def randomCombination(iterable,n):
    """
    Generator that yields a random index from iterable
    """

    i = 0
    pool = tuple(iterable)
    rng = range(len(pool))
    indices = random.sample(rng, n)
    while i < n:
        yield pool[indices[i]]
        i += 1


def getColorMaps(examples, n, doNotReplace=[0]):
    """
    Returns a list of colorMaps each of which fully determines how to transform a task where each item is (oldColor, newColor)

    Args:
        examples: List of task examples
        n: Desired number of colorMaps to generate
        doNotReplace: List of colors that should not be replaced

    Returns:
        List of color maps
    """

    colorsToReplace = getColorsToReplace(examples, doNotReplace)
    colorPallete = set(list(intToColor.keys())) - set(doNotReplace)
    k = len(colorsToReplace)
    maxExamples = int(perm(len(colorPallete), k))
    if perm(len(colorPallete), k) < n:
        print('WARNING: Can only generate {} examples for task:'.format(maxExamples))
        n = maxExamples
        # raise LessPermutationsThanRequiredExamples
    iterable = itertools.permutations(colorPallete, k)
    newColors = list(randomCombination(iterable, n))
    return [{oldColor:colorList[i] for i,oldColor in enumerate(colorsToReplace)} for colorList in newColors]

def colorPermuteExample(example, colorMap):
    """
    Transforms example by replacing its colors based on colorMap

    Args:
        example: Input, output pair
        colorMap: Map of colors where the key is the old color and the corresponding value is the new color to replace it by

    Returns:
        Transformed example with colors replaced according to colorMap
    """

    example = copy.deepcopy(example)
    inputGrid, outputGrid = example[0][0], example[1]
    return ((replaceColors(inputGrid, colorMap),), replaceColors(outputGrid, colorMap))

def generateFromFrontier(frontier, n):
    """
    Finds and permutes the non-special colors in a task to generate new tasks that the same programs will solve.

    Args:
        frontier: A single task frontier
        n: An integer indicating how many new tasks to generate. If n is greater than the number
        of total non-special color permutations of a task, then returns as many tasks as there are non-special color permutations.

    Returns:
        List of tasks that have been generate based on the frontier task by permuting the non-special colors 
    """

    programs, task = [str(frontierEntry.program) for frontierEntry in copy.deepcopy(frontier.entries)], copy.deepcopy(frontier.task)
    examples = task.examples
    doNotReplace = set([0])
    for color in colorToInt.keys():
        if color in ' '.join(programs):
            doNotReplace.add(colorToInt[color])
    print('Do not replace {} from task {}'.format(', '.join([intToColor[c] for c in list(doNotReplace)]), task))
    colorMaps = getColorMaps(examples, n, doNotReplace)
    if len(colorMaps) < n:
        print(task)
    n = min(len(colorMaps), n)
    generatedTasks = []
    for t in range(n):
        newTask = copy.deepcopy(task)
        newTask.name = '{}_{}'.format(t, task.name)
        for example in examples:
            newExample = colorPermuteExample(example, colorMaps[t])
            newTask.examples.append(newExample)
        generatedTasks.append(newTask)
    return generatedTasks

def expandFrontier(frontiers, n):
    """ 
    Expands each frontier in frontier by permuting the non-special colors.

    Args:
        frontiers: A list of frontiers where each frontier corresponds to one solved task
        n: An integer indicating how many color permutations to return for each frontier. If n is greater than the number
        of total color permutations of an task, then returns the all the color permutations

    Returns:
        A list of expanded frontiers where each frontier is at most n times the size it was originally 
    """

    expandedFrontiers = []
    for frontier in frontiers:
        try:
            expandedTasks = generateFromFrontier(frontier, n)
        except LessPermutationsThanRequiredExamples:
            print('LessPermutationsThanRequiredExamples')

        for task in expandedTasks:
            newFrontier = copy.deepcopy(frontier)
            newFrontier.task = task
            expandedFrontiers.append(newFrontier)
    expandedFrontiers += frontiers
    print('Expanded Frontier has length: {}'.format(len(expandedFrontiers)))
    return expandedFrontiers