

def g():
    yield 1
    yield 2

def f():
    yield 3
    return g()

print list(f())
