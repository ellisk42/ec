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
        mass = random.choice(masses)

        heights = { t.minimumHeight for t in TowerTask.tasks }
        lengths = { t.minimumLength for t in TowerTask.tasks }
        areas = { t.minimumArea for t in TowerTask.tasks }
        staircases = { t.maximumStaircase for t in TowerTask.tasks }

        # Find the largest perturbation that this power can withstand
        perturbations = sorted({ t.perturbation for t in TowerTask.tasks }, reverse=True)        
        for perturbation in perturbations:
            result = TowerTask.evaluateTower(p, perturbation)
            
            possibleHeightThresholds = { h for h in heights if result.height >= h }
            possibleLengthThresholds = { l for l in lengths if result.length >= l }
            possibleAreaThresholds = { a for a in areas if result.area >= a }
            possibleStaircases = { s for s in staircases if result.staircase <= s }
            
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
            else: return None

        return None

def evaluateArches(ts):
    arches = [
        "(do (do (left 1x3) (do (right 1x3) 3x1)) (right (right (right (do (left 1x3) (do (right 1x3) 3x1))))))",
       "(do 1x4 (do (left 1x4) 4x1))",
       "(do (right 1x4) (do (left (left 1x4)) 4x1))",
        "(do (do (right 1x4) (do (left (left 1x4)) 4x1)) (right (right (right (right (do (right 1x4) (do (left (left 1x4)) 4x1)))))))",
    ]
    towers = []

    for a in arches:
        print("Evaluating arch:")
        print(a)
        print()
        a = Program.parse(a).evaluate([])
        towers.append(tuple(centerTower(a)))
        os.system("python towers/visualize.py '%s' %f"%(a, 8))

        for t in ts:
            print(t, end=' ')
            print(t.logLikelihood(Primitive(str(a),None,a)),t.logLikelihood(Primitive(str(a*2),None,a*2)), end=' ')
            print()
        print()
        print()

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
    towers = [ np.concatenate(ts + [np.zeros(size, dtype=tp)]*(m - len(ts)), axis=1)
               for ts in towers ]
    towers = np.concatenate(towers, axis=0)
    Image.fromarray(towers).convert('RGB').save(name)

def bruteForceTower_(size):
    MAXIMUMWIDTH = 2
    if size == 0:
        yield []
        return

    moves = [ (x + dx,w,h)
              for p in primitives if 'x' in p.name
              for x,w,h in p.value
              for dx in range(-MAXIMUMWIDTH,MAXIMUMWIDTH+1) ]
    for b in moves:
        for s in bruteForceTower_(size - 1):
            yield [b] + s
def bruteForceTower(size):
    for s in range(1,size + 1):
        for t in bruteForceTower_(s):
            yield t
def bruteForceBaseline(tasks):
    from towers.tower_common import TowerWorld
    from PIL import Image
    towers = set([tuple(centerTower(t)) for t in bruteForceTower(4)])
    print("Generated",len(towers),"towers")
    for t in towers:
        gotHit = False
        for task in tasks:
            ll = task.logLikelihood(Primitive(str(t),None,t))
            if valid(ll):
                print("Hit",task,"w/")
                print(t)
                print() 
                # image = TowerWorld().draw(t)
                # Image.fromarray(image).convert('RGB').save("/tmp/towerBaseline.png")
                # os.system("feh /tmp/towerBaseline.png")
                gotHit = True
                break
        if gotHit:
            tasks = [task_ for task_ in tasks if not task == task_ ]
                
    import sys
    sys.exit(0)
    
        

    

if __name__ == "__main__":
    initializeTowerCaching()
    
    g0 = Grammar.uniform(primitives)
    tasks = makeTasks()
    test, train = testTrainSplit(tasks, 100./len(tasks))
    eprint("Split %d/%d test/train"%(len(test),len(train)))
    # evaluateArches(train)
    if False: bruteForceBaseline(train)

    generator = ecIterator(g0, train,
                           testingTasks=test,
                           outputPrefix="experimentOutputs/tower",
                           solver="python",
                           **commandlineArguments(
                               featureExtractor=TowerFeatureExtractor,
                               CPUs=numberOfCPUs(),
                               helmholtzRatio=0.5,
                               iterations=5,
                               a=3,
                               structurePenalty=1,
                               pseudoCounts=10,
                               topK=10,
                               maximumFrontier=10**4))
    # list of list of towers, one for each iteration
    towers = []
    for result in generator:
        newTowers = { tuple(centerTower(frontier.bestPosterior.program.evaluate([])))
                      for frontier in result.taskSolutions.values() if not frontier.empty }
        towers.append(sorted(newTowers))
        exportTowers(towers, 'experimentOutputs/uniqueTowers.png')
