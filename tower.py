from ec import *

from towerPrimitives import primitives
from makeTowerTasks import *

import os
import random

class TowerFeatureExtractor(HandCodedFeatureExtractor):
    def _featuresOfProgram(self, p, _):
        # [perturbation, mass, height, length, area]
        p = p.evaluate([])
        mass = sum(w*h for _,w,h in p)

        masses = { t.maximumMass for t in TowerTask.tasks if mass <= t.maximumMass }
        if len(masses) == 0: return None
        mass = random.choice(list(masses))

        heights = { t.minimumHeight for t in TowerTask.tasks }
        lengths = { t.minimumLength for t in TowerTask.tasks }
        areas = { t.minimumArea for t in TowerTask.tasks }

        # Find the largest perturbation that this power can withstand
        perturbations = sorted({ t.perturbation for t in TowerTask.tasks }, reverse = True)        
        for perturbation in perturbations:
            result = TowerTask.evaluateTower(p, perturbation)
            
            possibleHeightThresholds = { h for h in heights if result.height >= h }
            possibleLengthThresholds = { l for l in lengths if result.length >= l }
            possibleAreaThresholds = { a for a in areas if result.area >= a }
            
            if len(possibleHeightThresholds) > 0 and \
               len(possibleLengthThresholds) > 0 and \
               len(possibleAreaThresholds) > 0:
                if result.stability > TowerTask.STABILITYTHRESHOLD:
                    return [perturbation,
                            mass,
                            random.choice(list(possibleHeightThresholds)),
                            random.choice(list(possibleLengthThresholds)),
                            random.choice(list(possibleAreaThresholds))]
            else: return None

        return None

def evaluateArches(ts):
    arches = [
        # "(do unit (do unit unit))",
       "(do 1x4 (do (left 1x4) 4x1))",
       "(do (right 1x4) (do (left (left 1x4)) 4x1))",
        "(do (do (right 1x4) (do (left (left 1x4)) 4x1)) (right (right (right (right (do (right 1x4) (do (left (left 1x4)) 4x1)))))))",
#        "(do (left tallVertical) (do (right tallVertical) wideHorizontal))",
#        "(do (left tallVertical) (do (right tallVertical) horizontalBrick))",
        # "(do (left tallVertical) (do (right tallVertical) tallVertical))",
        # "(do tallVertical tallVertical)",
        # "(do tallVertical horizontalBrick)",
        # "(do tallVertical (do tallVertical tallVertical))"
    ]
    towers = []

    for a in arches:
        print "Evaluating arch:"
        print a
        print
        a = Program.parse(a).evaluate([])
        towers.append(tuple(centerTower(a)))
        #os.system("python towers/visualize.py '%s' %f"%(a, 8))

        for t in ts:
            print t,
            print t.logLikelihood(Primitive(str(a),None,a)),t.logLikelihood(Primitive(str(a*2),None,a*2)),
            print
        print
        print

    exportTowers([towers[:1],towers[:2],towers], "arches.png")
    import sys
    sys.exit()

def exportTowers(towers, name):
    from PIL import Image
    from towers.tower_common import TowerWorld

    m = max(len(t) for t in towers)
    towers = [ [ TowerWorld().draw(t) for t in ts ]
               for ts in towers ]
    
    size = towers[0][0].shape
    tp = towers[0][0].dtype
    towers = [ np.concatenate(ts + [np.zeros(size, dtype = tp)]*(m - len(ts)), axis = 1)
               for ts in towers ]
    towers = np.concatenate(towers, axis = 0)
    Image.fromarray(towers).convert('RGB').save(name)
    

if __name__ == "__main__":
    g0 = Grammar.uniform(primitives)
    tasks = makeTasks()
    test, train = testTrainSplit(tasks, 100./len(tasks))
    eprint("Split %d/%d test/train"%(len(test),len(train)))
    # evaluateArches(train)

    generator = ecIterator(g0, train,
                           testingTasks = test,
                           outputPrefix = "experimentOutputs/tower",
                           solver = "python",
                           **commandlineArguments(
                               featureExtractor = TowerFeatureExtractor,
                               CPUs = numberOfCPUs(),
                               helmholtzRatio = 0.5,
                               iterations = 5,
                               a = 3,
                               structurePenalty = 1,
                               pseudoCounts = 10,
                               topK = 10,
                               maximumFrontier = 10**4))
    # list of list of towers, one for each iteration
    towers = []
    for result in generator:
        newTowers = { tuple(centerTower(frontier.bestPosterior.program.evaluate([])))
                      for frontier in result.taskSolutions.values() if not frontier.empty }
        towers.append(sorted(list(newTowers)))
        exportTowers(towers, 'uniqueTowers.png')
