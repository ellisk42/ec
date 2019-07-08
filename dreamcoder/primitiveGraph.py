from dreamcoder.program import *

def graphPrimitives(result, prefix, view=False):
    try:
        from graphviz import Digraph
    except:
        eprint("You are missing the graphviz library - cannot graph primitives!")
        return
    

    primitives = { p
                   for g in result.grammars
                   for p in g.primitives
                   if p.isInvented }
    age = {p: min(j for j,g in enumerate(result.grammars) if p in g.primitives) + 1
           for p in primitives }



    ages = set(age.values())
    age2primitives = {a: {p for p,ap in age.items() if a == ap }
                      for a in ages}

    def lb(s,T=20):
        s = s.split()
        l = []
        n = 0
        for w in s:
            if n + len(w) > T:
                l.append("<br />")
                n = 0
            n += len(w)
            l.append(w)
        return " ".join(l)

    nameSimplification = {
        "fix1": 'Y',
        "tower_loopM": "for",
        "tower_embed": "get/set",
        "moveHand": "move",
        "reverseHand": "reverse",
        "logo_DIVA": '/',
        "logo_epsA": 'ε',
        "logo_epsL": 'ε',
        "logo_IFTY": '∞',
        "logo_forLoop": "for",
        "logo_UA": "2π",
        "logo_FWRT": "move",
        "logo_UL": "1",
        "logo_SUBA": "-",
        "logo_ZL": "0",
        "logo_ZA": "0",
        "logo_MULL": "*",
        "logo_MULA": "*",
        "logo_PT": "pen-up",
        "logo_GETSET": "get/set"
    }

                
    name = {}
    simplification = {}
    depth = {}
    def getName(p):
        if p in name: return name[p]
        children = {k: getName(k)
                    for _,k in p.body.walk()
                    if k.isInvented}
        simplification_ = p.body
        for k,childName in children.items():
            simplification_ = simplification_.substitute(k, Primitive(childName,None,None))
        for original, simplified in nameSimplification.items():
            simplification_ = simplification_.substitute(Primitive(original,None,None),
                                                         Primitive(simplified,None,None))
        name[p] = "f%d"%len(name)
        simplification[p] = name[p] + '=' + lb(prettyProgram(simplification_, Lisp=True))
        depth[p] = 1 + max([depth[k] for k in children] + [0])
        return name[p]

    for p in primitives:
        getName(p)

    depths = {depth[p] for p in primitives}
    depth2primitives = {d: {p for p in primitives if depth[p] == d }
                        for d in depths}

    englishDescriptions = {"#(lambda (lambda (map (lambda (index $0 $2)) (range $0))))":
                           "Prefix",
                           "#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0))))))":
                           "Append",
                           "#(lambda (cons LPAREN (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) (cons RPAREN empty) $0)))":
                           "Enclose w/ parens",
                           "#(lambda (unfold $0 (lambda (empty? $0)) (lambda (car $0)) (lambda (#(lambda (lambda (fold $1 $1 (lambda (lambda (cdr (if (char-eq? $1 $2) $3 $0))))))) $0 SPACE))))":
                           "Abbreviate",
                           "#(lambda (lambda (fold $1 $1 (lambda (lambda (cdr (if (char-eq? $1 $2) $3 $0)))))))":
                           "Drop until char",
                           "#(lambda (lambda (fold $1 $1 (lambda (lambda (if (char-eq? $1 $2) empty (cons $1 $0)))))))":
                           "Take until char",
                           "#(lambda (lambda (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) (cons $0 $1))))":
                           "Append char",
                           "#(lambda (lambda (map (lambda (if (char-eq? $0 $1) $2 $0)))))":
                           "Substitute char",
                           "#(lambda (lambda (length (unfold $1 (lambda (char-eq? (car $0) $1)) (lambda ',') (lambda (cdr $0))))))":
                           "Index of char",
                           "#(lambda (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) $0 STRING))":
                           "Append const",
                           "#(lambda (lambda (fold $1 $1 (lambda (lambda (fold $0 $0 (lambda (lambda (cdr (if (char-eq? $1 $4) $0 (cons $1 $0)))))))))))":
                           "Last word",
                           "#(lambda (lambda (cons (car $1) (cons '.' (cons (car $0) (cons '.' empty))))))":
                           "Abbreviate name",
                           "#(lambda (lambda (cons (car $1) (cons $0 empty))))":
                           "First char+char",
                           "#(lambda (#(lambda (lambda (fold $0 $1 (lambda (lambda (cons $1 $0)))))) (#(lambda (lambda (fold $1 $1 (lambda (lambda (fold $0 $0 (lambda (lambda (cdr (if (char-eq? $1 $4) $0 (cons $1 $0))))))))))) STRING (index (length (cdr $0)) $0)) $0))":
                           "Ensure suffix"
                           
    }

    def makeUnorderedGraph(fn):
        g = Digraph()
        g.graph_attr['rankdir'] = 'LR'

        for p in primitives:
            g.node(getName(p),
                   label="<%s>"%simplification[p])
        for p in primitives:
            children = {k
                        for _,k in p.body.walk()
                        if k.isInvented}
            for k in children:
                g.edge(name[k],name[p])
        try:
            g.render(fn,view=view)
            eprint("Exported primitive graph to",fn)
        except:
            eprint("Got some kind of error while trying to render primitive graph! Did you install graphviz/dot?")

        

    def makeGraph(ordering, fn):
        g = Digraph()
        g.graph_attr['rankdir'] = 'RL'

        if False:
            with g.subgraph(name='cluster_0') as sg:
                sg.graph_attr['rank'] = 'same'
                sg.attr(label='Primitives')
                for j, primitive in enumerate(result.grammars[-1].primitives):
                    if primitive.isInvented: continue
                    sg.node("primitive%d"%j, label=str(primitive))

        for o in sorted(ordering.keys()):
            with g.subgraph(name='cluster_%d'%o) as sg:
                sg.graph_attr['rank'] = 'same'
                #sg.attr(label='Depth %d'%o)
                for p in ordering[o]:
                    if str(p) in englishDescriptions:
                        thisLabel = '<<font face="boldfontname"><u>%s</u></font><br />%s>'%(englishDescriptions[str(p)],simplification[p])
                    else:
                        eprint("WARNING: Do not have an English description of:\n",p)
                        eprint()
                        thisLabel = "<%s>"%simplification[p]
                    sg.node(getName(p),
                            label=thisLabel)

            for p in ordering[o]:
                children = {k
                            for _,k in p.body.walk()
                            if k.isInvented}
                for k in children:
                    g.edge(name[k],name[p])

        eprint("Exporting primitive graph to",fn)
        try:
            g.render(fn,view=view)
        except Exception as e:
            eprint("Got some kind of error while trying to render primitive graph! Did you install graphviz/dot?")
            print(e)
        
        

    makeGraph(depth2primitives,prefix+'depth.pdf')
    makeUnorderedGraph(prefix+'unordered.pdf')
    #makeGraph(age2primitives,prefix+'iter.pdf')
