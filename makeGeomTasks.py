# coding: utf8

from task import Task
from geomPrimitives import tcanvas
import png
import os

rootdir = "./data/geom/"


def fileToArray(fname):
    r = png.Reader(filename=fname)
    array = [[y for y in x[3::4]] for x in r.read()[2]]
    flatten = [item for sublist in array for item in sublist]
    return flatten


def pretty_string(shape, size):
    out = ""
    nl = "\n"
    out += "╭"
    out += "─"*(size*2)
    out += "╮"
    out += nl
    for j in range(size):
        out += "│"
        for i in range(size):
            if int(shape[j*size + (i % size)]) == 0:
                out += "░░"
            else:
                out += "██"
        out += "│"
        out += nl
    out += "╰"
    out += "─"*(size*2)
    out += "╯"
    out += nl
    return out


def pretty_print(shape, size):
    print (pretty_string(shape, size))


def makeTasks():
    problems = []

    def problem(n, examples, needToTrain=False):
        outputType = tcanvas
        task = Task(n,
                    outputType,
                    [((), y) for _, y in examples])
        task.mustTrain = needToTrain
        problems.append(task)

    for _, _, files in os.walk(rootdir):
        for f in files:
            if f.endswith(".png"):
                problem(f,
                        [([], fileToArray(rootdir + '/' + f))],
                        needToTrain=True)

    return problems


if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        print t.name
        print t.request
        x, y = t.examples[0]
        pretty_print(y, 64)
        print
