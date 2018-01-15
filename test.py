
def f():
    x = []
    def g():
        print "x = ",x
    g()
    x = 9
    g()
    for j in range(5):
        x = j
        g()



f()
