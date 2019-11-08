try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.dreamcoder import *

from dreamcoder.grammar import *
from dreamcoder.program import *

import numpy as np

from dreamcoder.type import *

tobject = baseType("object")
tvector = baseType("vector")
tfield = baseType("field")

Primitive("get-field",arrow(tobject,tfield,tvector),None)
Primitive("mass",arrow(tobject,treal),None)
Primitive("*v",arrow(treal,tvector,tvector),None)
Primitive("/v",arrow(tvector,treal,tvector),None)
Primitive("dp",arrow(tvector,tvector,treal),None)
Primitive("-v",arrow(tvector,tvector,tvector),None)
Primitive("+v",arrow(tvector,tvector,tvector),None)
Primitive("get-position",arrow(tobject,tvector),None)
Primitive("get-velocity",arrow(tobject,tvector),None)
Primitive("position",tfield,None)
Primitive("velocity",tfield,None)
Primitive("normalize",arrow(tvector,tvector),None)
Primitive("yhat",tvector,None)
Primitive("vector-length",arrow(tvector,treal),None)
Primitive("sq",arrow(treal,treal),None)

class Particle:
    def __init__(self,m,x,v):
        self.m = m
        self.x = x
        self.v = v

    def json(self):
        return {"mass": self.m,
                "position": list(self.x),
                "velocity": list(self.v)}

    def __str__(self):
        return "Particle(mass=%f, x=%s, v=%s)"%(self.m,self.x,self.v)
    def __repr__(self):
        return str(self)

    def step(self,f,dt):
        return Particle(self.m,
                        self.x + self.v*dt,
                        self.v + f/self.m*dt)

class Vignette():
    def __init__(self, *trajectories):
        "trajectory: list of PhysicsObject. Should be the same object, so mass should be constant"
        self.trajectories = trajectories
        for t in trajectories:
            m = t[0].m
            for j,o in enumerate(t[1:]):
                assert o.m == m

        self.l = len(self.trajectories[0])
        for t in trajectories:
            assert len(t) == self.l

    def __len__(self): return self.l

    def visualize(self):
        import matplotlib.pyplot as plot
        plot.figure()
        colors = [[1.,0,0],
                  [0,1.,0.],
                  [0.,0.,1.]] #['r','b','g']
        colors = ['r','b','g']
        for t in self.trajectories:
            xs = []
            ys = []
            for o in t:
                xs.append(o.x[0])
                ys.append(o.x[1])
            plot.scatter(xs,ys,
                         c=colors[0])
            colors = colors[1:]
        plot.show()


def freefallVignette():
    trajectories = []
    for _ in range(1):
        m = random.random()
        x0 = np.array([random.random(), random.random()])*2 - np.array([1,1])
        v0 = np.array([random.random(), random.random()])*2 - np.array([1,1])
        v0 = v0*0.2
        p = Particle(m,x0,v0)
        trajectory = []
        for _ in range(20):
            trajectory.append(p)
            p = p.step(np.array([0.,-0.5])*p.m,
                       1)
        trajectories.append(trajectory)
    return Vignette(*trajectories)

def spring(k, n):
    trajectories = []
    for _ in range(n):
        m = random.random()
        x0 = np.array([random.random(), random.random()])*2 - np.array([1,1])
        v0 = np.array([random.random(), random.random()])*2 - np.array([1,1])
        v0 = v0*0.2
        p = Particle(m,x0,v0)
        trajectory = []
        for _ in range(100):
            trajectory.append(p)
            f = -k * p.x
            p = p.step(f, 0.01)
        trajectories.append(trajectory)
    return Vignette(*trajectories)
        
    
def airResistance(k, n):
    trajectories = []
    for _ in range(n):
        m = random.random()
        x0 = np.array([random.random(), random.random()])*2 - np.array([1,1])
        v0 = np.array([random.random(), random.random()])*2 - np.array([1,1])
        v0 = v0*0.2
        p = Particle(m,x0,v0)
        trajectory = []
        for _ in range(10):
            trajectory.append(p)
            f = -k * (p.v*p.v).sum()**0.5
            f = f*p.v/(p.v*p.v).sum()**0.5
            p = p.step(f,
                       0.1)
        trajectories.append(trajectory)
    return Vignette(*trajectories)

def gravity(g):
    trajectories = [[],[]]
    m = random.random()
    def makeParticle(i):
        x0 = np.array([-2 if i == 0 else 2,0.])*10
        v0 = np.array([0., 0 if i == 0 else -2])
        v0 = v0*.9
        m = 100. if i == 0 else 0.1
        p = Particle(m,x0,v0)
        return p
    objects = [makeParticle(0),makeParticle(1)]
    for _ in range(30000):
        newObjects = []
        for i in range(len(objects)):
            f = 0
            for j in range(len(objects)):
                if i == j: continue
                r = objects[i].x - objects[j].x
                rl = (r*r).sum()**0.5
                rh = r/rl
                r2 = rl*rl
                f -= g*rh*objects[i].m*objects[j].m/r2

            newObjects.append(objects[i].step(f,0.1))
        objects = newObjects
        for i in range(len(objects)):
            trajectories[i].append(objects[i])
    return Vignette(*trajectories)

                

def makeTasks(namePrefix, vignettes):
    for v in vignettes:
        assert len(v.trajectories) == len(vignettes[0].trajectories)
        eprint(namePrefix)
        v.visualize()
        
    outputObjects = len(v.trajectories)
    velocityExamples = []
    deltaVelocityExamples = []
    for v in vignettes:
        for i in range(len(v.trajectories)):
            # Predict velocity of object i from all of the other objects 
            for t in range(1, len(v)): # for each time step
                y = v.trajectories[i][t].v
                xs = [v.trajectories[i][t - 1].json()] + [v.trajectories[j][t - 1].json()
                                                          for j in range(len(v.trajectories))
                                                          if j != i]
                velocityExamples.append((xs,list(y)))
                y = y - v.trajectories[i][t - 1].v
                deltaVelocityExamples.append((xs,list(y)))

    tasks = []
    for suffix, examples in [("velocity", velocityExamples),
                             ("delta", deltaVelocityExamples)]:
        t = Task("%s-%s"%(namePrefix, suffix),
                 arrow(*([tobject]*(outputObjects) + [tvector])),
                 [])
        t.specialTask = ("physics",
                         {"parameterPenalty": 20,
                          "maxParameters": 1,
                          "temperature": 1.,
                          "restarts": 1,
                          "steps": 200,
                          "lossThreshold": 0.00005,
                          "examples": examples})
        tasks.append(t)
    return tasks


def physicsTasks():
    tasks = makeTasks("freefall",
                   [freefallVignette()]) + \
                   makeTasks("spring",[spring(1,3)]) + \
                   makeTasks("viscous",[airResistance(0.1,3)]) + \
                   makeTasks("gravity",[gravity(2)])
    # subsample examples
    numberOfExamples = 1000
    for t in tasks:
        examples = t.specialTask[1]["examples"]
        if len(examples) > numberOfExamples:
            t.specialTask[1]["examples"] = examples[0:len(examples):len(examples)//numberOfExamples]
    return tasks

def groundTruthSolutions():
    solutions = ["(lambda (o) (+v (get-velocity o) (*v REAL yhat)))",
                 "(lambda (o) (+v (get-velocity o) (*v REAL (/v (get-velocity o) (mass o)))))",
                 "(lambda (o) (+v (get-velocity o) (*v (/. REAL (mass o)) (get-position o))))"]
    return [Invented(Program.parseHumanReadable(s)) for s in solutions ]

def physics_options(parser):
    pass

if __name__ == "__main__":
    g = Grammar.uniform([Program.parse(p)
                         for p in ["mass","get-position","get-velocity","normalize","vector-length",
                                   "sq",
                                   "yhat",
                                   "-v","+v","*v","/v","dp",
                                   "/.","+.","*.","REAL"] ] + groundTruthSolutions())
    arguments = commandlineArguments(
        iterations=6,
        helmholtzRatio=0.0,
        topK=2,
        maximumFrontier=5,
        structurePenalty=1.5,
        a=3,
        activation="tanh",
        CPUs=numberOfCPUs(),
        featureExtractor=None,
        useRecognitionModel=False,
        pseudoCounts=30.0,
        extras=physics_options)

    tasks = physicsTasks()

    # generator = ecIterator(g, tasks,
    #                        outputPrefix="experimentOutputs/physics",
    #                        evaluationTimeout=0.01,
    #                        **arguments)
    # for result in generator:
    #     pass

    # assert False

    
    def showLikelihood(e):
        e = Program.parse(e)
        while e.numberOfFreeVariables > 0:
            e = Abstraction(e)
        t = e.infer()
        assert not t.isPolymorphic
        eprint(e," : ",t,"\n\t",g.logLikelihood(t,e))
    forces = ["(*v REAL (get-position $0))",
              "(*v REAL yhat)",
              "(+v (*v REAL (get-position $0)) (*v REAL yhat))",
              "(*v (/. REAL (dp (get-position $0) (get-position $0))) (normalize (get-position $0)))",
              "(*v REAL (*v (vector-length (-v (get-position $0) (get-position $1))) (normalize (-v (get-position $0) (get-position $1)))))",
              """(*v 
               (*. REAL (/. (*. (mass $0) (mass $1)) (sq (vector-length (-v (get-position $0) (get-position $1))))))
               (normalize (-v (get-position $0) (get-position $1))))""",
              """(*v 
               (*. REAL (/. (*. (mass $0) (mass $1)) (dp (-v (get-position $0) (get-position $1)) (-v (get-position $0) (get-position $1)))))
               (normalize (-v (get-position $0) (get-position $1))))"""]

    
    for f in forces:
        dv = "(/v %s (mass $0))"%f
        showLikelihood(dv)
        dt = "$%d"%(Program.parse(dv).numberOfFreeVariables)
        dv = "(*v %s %s)"%(dt,dv)
        showLikelihood(dv)
        dv = "(+v (get-velocity $0) %s)"%dv
        showLikelihood(dv)

        eprint()
        eprint()
