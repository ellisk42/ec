from ec import *

from towerPrimitives import primitives, executeTower
from makeTowerTasks import *
from listPrimitives import bootstrapTarget

import os
import random
import time
import datetime


class TowerFeatureExtractor(HandCodedFeatureExtractor):
    def _featuresOfProgram(self, p, _):
        # [perturbation, mass, height, length, area]
        p = executeTower(p)
        mass = sum(w * h for _, w, h in p)

        masses = {
            t.maximumMass for t in TowerTask.tasks if mass <= t.maximumMass}
        if len(masses) == 0:
            return None
        mass = random.choice(masses)

        heights = {t.minimumHeight for t in TowerTask.tasks}
        lengths = {t.minimumLength for t in TowerTask.tasks}
        areas = {t.minimumArea for t in TowerTask.tasks}
        staircases = {t.maximumStaircase for t in TowerTask.tasks}

        # Find the largest perturbation that this power can withstand
        perturbations = sorted(
            {t.perturbation for t in TowerTask.tasks}, reverse=True)
        for perturbation in perturbations:
            result = TowerTask.evaluateTower(p, perturbation)

            possibleHeightThresholds = {
                h for h in heights if result.height >= h}
            possibleLengthThresholds = {
                l for l in lengths if result.length >= l}
            possibleAreaThresholds = {a for a in areas if result.area >= a}
            possibleStaircases = {
                s for s in staircases if result.staircase <= s}

            if len(possibleHeightThresholds) > 0 and \
               len(possibleLengthThresholds) > 0 and \
               len(possibleStaircases) > 0 and \
               len(possibleAreaThresholds) > 0:
                if result.stability > TowerTask.STABILITYTHRESHOLD:
                    return [perturbation,
                            mass,
                            random.choice(possibleHeightThresholds),
                            random.choice(possibleLengthThresholds),
                            random.choice(possibleAreaThresholds),
                            random.choice(possibleStaircases)]
            else:
                return None

        return None


def evaluateArches(ts):
    arches = [
        "(lambda (left (1x3 (right (right (right (1x3 (left (3x1 $0)))))))))"
        #  "(do (do (left 1x3) (do (right 1x3) 3x1)) (right (right (right (do (left 1x3) (do (right 1x3) 3x1))))))",
        # "(do 1x4 (do (left 1x4) 4x1))",
        # "(do (right 1x4) (do (left (left 1x4)) 4x1))",
        #  "(do (do (right 1x4) (do (left (left 1x4)) 4x1)) (right (right (right (right (do (right 1x4) (do (left (left 1x4)) 4x1)))))))",
    ]
    towers = []

    for a in arches:
        print("Evaluating arch:")
        print(a)
        print()
        a = Program.parse(a)
        towers.append(tuple(centerTower(a)))
        os.system("python towers/visualize.py '%s' %f" % (a, 4))

        for t in ts:
            print(t, end=' ')
            print(t.logLikelihood(Primitive(str(a), None, a)),
                  t.logLikelihood(Primitive(str(a * 2), None, a * 2)), end=' ')
            print()
        print()
        print()

    exportTowers(towers, "arches.png")
    import sys
    sys.exit()



def bruteForceTower_(size):
    MAXIMUMWIDTH = 2
    if size == 0:
        yield []
        return

    moves = [(x + dx, w, h)
             for p in primitives if 'x' in p.name
             for x, w, h in p.value
             for dx in range(-MAXIMUMWIDTH, MAXIMUMWIDTH + 1)]
    for b in moves:
        for s in bruteForceTower_(size - 1):
            yield [b] + s


def bruteForceTower(size):
    for s in range(1, size + 1):
        yield from bruteForceTower_(s)


def bruteForceBaseline(tasks):
    from towers.tower_common import TowerWorld
    from PIL import Image
    towers = set([tuple(centerTower(t)) for t in bruteForceTower(4)])
    print("Generated", len(towers), "towers")
    for t in towers:
        gotHit = False
        for task in tasks:
            ll = task.logLikelihood(Primitive(str(t), None, t))
            if valid(ll):
                print("Hit", task, "w/")
                print(t)
                print()
                # image = TowerWorld().draw(t)
                # Image.fromarray(image).convert('RGB').save("/tmp/towerBaseline.png")
                # os.system("feh /tmp/towerBaseline.png")
                gotHit = True
                break
        if gotHit:
            tasks = [task_ for task_ in tasks if not task == task_]

    import sys
    sys.exit(0)





if __name__ == "__main__":
    from towers.tower_common import exportTowers
    initializeTowerCaching()

    g0 = Grammar.uniform(primitives + 
                         [Primitive(str(j), tint, j) for j in range(2, 5)])

    tasks = makeTasks()
    test, train = testTrainSplit(tasks, 1.) #50. / len(tasks))
    eprint("Split %d/%d test/train" % (len(test), len(train)))

    arguments = commandlineArguments(
        featureExtractor=TowerFeatureExtractor,
        CPUs=numberOfCPUs(),
        helmholtzRatio=0.5,
        iterations=5,
        a=3,
        structurePenalty=1,
        pseudoCounts=10,
        topK=10,
        maximumFrontier=10**4)
    evaluationTimeout = 0.005
    generator = ecIterator(g0, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/tower",
                           evaluationTimeout=evaluationTimeout,
                           solver="ocaml",
                           compressor="pypy",
                           **arguments)
    os.system("python towers/server.py KILL")
    time.sleep(1)
    os.system("python towers/server.py &")
    time.sleep(1)

    perturbations = {t.perturbation for t in train}

    timestamp = datetime.datetime.now().isoformat()
    os.system("mkdir -p experimentOutputs/towers/%s"%timestamp)
    
    for result in generator:
        iteration = len(result.learningCurve)
        newTowers = [tuple(centerTower(executeTower(frontier.sample().program)))
                     for frontier in result.taskSolutions.values() if not frontier.empty]
        fn = 'experimentOutputs/towers/%s/solutions_%d.png'%(timestamp,iteration)
        exportTowers(newTowers, fn)
        eprint("Exported solutions to %s\n"%fn)
