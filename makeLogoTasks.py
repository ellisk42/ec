# coding: utf8

from task import Task
from type import arrow
from logoPrimitives import turtle
import png
import os
import sys

rootdir = "./data/logo/"


def fileToArray(fname):
    r = png.Reader(filename=fname)
    array = [[y for y in x[3::4]] for x in r.read()[2]]
    flatten = [item for sublist in array for item in sublist]
    return flatten


def pretty_string(shape, size):
    out = ""
    nl = "\n"
    out += "╭"
    out += "─" * (size * 2)
    out += "╮"
    out += nl
    for j in range(size):
        out += "│"
        for i in range(size):
            if int(shape[j * size + (i % size)]) < 51:
                out += "  "
            elif int(shape[j * size + (i % size)]) < 102:
                out += "░░"
            elif int(shape[j * size + (i % size)]) < 153.6:
                out += "▒▒"
            elif int(shape[j * size + (i % size)]) < 204.8:
                out += "▓▓"
            else:
                out += "██"
        out += "│"
        out += nl
    out += "╰"
    out += "─" * (size * 2)
    out += "╯"
    out += nl
    return out


def pretty_print(shape, size):
    print((pretty_string(shape, size)))


def allTasks():
    return next(os.walk(rootdir))[1]


def makeTasks(subfolders, proto):
    problems = []

    if subfolders == ['all']:
        subfolders = allTasks()

    def problem(n, examples, highresolution, needToTrain=False):
        outputType = arrow(turtle, turtle)
        task = Task(n,
                    outputType,
                    [([0], y) for _, y in examples])
        task.mustTrain = needToTrain
        task.proto = proto
        task.specialTask = ("LOGO", {"proto": proto})
        task.highresolution = highresolution
        problems.append(task)

    for subfolder in subfolders:
        for _, subf, _ in os.walk(rootdir + subfolder):
            for subfl in subf:
                for _, _, files in os.walk(rootdir + subfolder + "/" + subfl):
                    for f in files:
                        if f.endswith("_l.png"):
                            fullPath = rootdir + subfolder + "/" + subfl + '/' + f
                            img1 = fileToArray(fullPath)
                            highresolution = fileToArray(fullPath.replace("_l.png", "_h.png"))
                            try:
                                problem(subfolder+"/"+subfl,
                                        [([], img1)],
                                        highresolution,
                                        needToTrain=True)
                            except FileNotFoundError:
                                problem(subfolder+"_"+f,
                                        [([], img1)],
                                        highresolution,
                                        needToTrain=True)
    return problems


if __name__ == "__main__":
    allTasks()
    if len(sys.argv) > 1:
        tasks = makeTasks(sys.argv[1:])
    else:
        tasks = makeTasks(['all'])
    for t in tasks:
        x, y = t.examples[0]
        pretty_print(y, 28)
        try:
            x, y = t.examples[1]
            pretty_print(y, 28)
        except IndexError:
            print("no NORM")
            pretty_print(y, 28)
        print()
