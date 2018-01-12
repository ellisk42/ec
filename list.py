from ec import *
from makeListProblems import makeTasks

primitives = [
    Primitive("+",arrow(tint,tint),lambda x: lambda y: x + y),
    Primitive("-",arrow(tint,tint),lambda x: lambda y: x - y),
    Primitive("sort",arrow(tlist(tint),tlist(tint)),lambda x: list(sorted(x))),
    Primitive("reverse",arrow(tlist(tint),tlist(tint)),lambda x: list(reversed(x))),
    Primitive("++",arrow(tlist(tint),tlist(tint),tlist(tint)),lambda x: lambda y: x + y),
    Primitive("singleton",arrow(tint,tlist(tint)),lambda x: [x]),
    Primitive("slice",arrow(tint,tint,tlist(tint),tlist(tint)),lambda x: lambda y: lambda l: l[x:y]),
    Primitive("len",arrow(tlist(tint),tint),len),
    Primitive("map",arrow(arrow(tint,tint),tlist(tint),tlist(tint)),map),
    Primitive("reduce",arrow(arrow(tint,tint), tint, tlist(tint), tint),
              lambda f: lambda x0: lambda l: reduce(f,l,x0)),
    Primitive("filter",arrow(arrow(tint,tbool), tlist(tint), tlist(tint)),
              lambda f: lambda x: filter(f,x)),
    Primitive("eq?",arrow(tint,tint,tbool), lambda x: lambda y: x == y),
    Primitive("mod",arrow(tint,tint,tint), lambda x: lambda y: x%y),
    Primitive("not",arrow(tbool,tbool), lambda x: not x),
    Primitive("gt?",arrow(tint,tint,tbool), lambda x: lambda y: x > y)    
] + [ Primitive(str(j),tint,j) for j in range(10) ]

if __name__ == "__main__":
    tasks = makeTasks()
    print "Got",len(tasks),"list tasks"
