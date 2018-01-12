from ec import *
from makeStringTransformationProblems import makeTasks


primitives = [
    Primitive("0",tint,0),
    Primitive("incr",arrow(tint,tint),lambda x: x + 1),
    Primitive("decr",arrow(tint,tint),lambda x: x - 1),
    Primitive("emptyString",tstring,""),
    Primitive("lowercase",arrow(tstring,tstring), lambda x: x.lower()),
    Primitive("uppercase",arrow(tstring,tstring), lambda x: x.upper()),
    Primitive("capitalize",arrow(tstring,tstring), lambda x: x.capitalize()),
    Primitive("++",arrow(tstring,tstring,tstring), lambda x: lambda y: x + y),
    Primitive("','", tstring, ","),
    Primitive("' '", tstring, " "),
    Primitive("'<'", tstring, "<"),
    Primitive("'>'", tstring, ">"),
    Primitive("'.'", tstring, "."),
    Primitive("'@'", tstring, "@"),
    Primitive("slice", arrow(tint,tint,tstring,tstring),
              lambda x: lambda y: lambda s: s[x:y]),
    Primitive("nth", arrow(tint, tlist(tstring), tstring),
              lambda n: lambda ss: s[n]),
    Primitive("map", arrow(arrow(tstring,tstring), tlist(tstring), tlist(tstring)),
              lambda f: lambda l: map(f,l)),
    Primitive("find", arrow(tstring, tstring, tint),
              lambda pattern: lambda s: s.index(pattern)),
    Primitive("replace", arrow(tstring, tstring, tstring, tstring),
              lambda original: lambda replacement: lambda target: target.replace(original, replacement)),
    Primitive("split", arrow(tstring, tstring, tlist(tstring)),
              lambda delimiter: lambda s: s.split(delimiter)),
    Primitive("join", arrow(tstring, tlist(tstring), tstring),
              lambda delimiter: lambda ss: delimiter.join(ss))
]

if __name__ == "__main__":
    tasks = makeTasks()
    for t in tasks:
        t.features = [1]
        t.cache = False
    print "Generated",len(tasks),"tasks"

    explorationCompression(primitives, tasks,
                           frontierSize = 10**4,
                           iterations = 3,
                           pseudoCounts = 10.0)
