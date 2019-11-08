from dreamcoder.program import Primitive
from dreamcoder.grammar import Grammar
from dreamcoder.type import arrow, tpregex
from string import printable
from pregex import pregex


# evaluation to regular regex form. then I can unflatten using Luke's stuff.


def _kleene(x): return pregex.KleeneStar(x, p=0.25)


def _plus(x): return pregex.Plus(x, p=0.25)


def _maybe(x): return pregex.Maybe(x)


# maybe should be reversed#"(" + x + "|" + y + ")"
def _alt(x): return lambda y: pregex.Alt([x, y])


def _concat(x): return lambda y: pregex.Concat([x, y])  # "(" + x + y + ")"


#For sketch:
def _kleene_5(x): return pregex.KleeneStar(x)

def _plus_5(x): return pregex.Plus(x)


disallowed = [
    ("#", "hash"),
    ("!", "bang"),
    ("\"", "double_quote"),
    ("$", "dollar"),
    ("%", "percent"),
    ("&", "ampersand"),
    ("'", "single_quote"),
    (")", "left_paren"),
    ("(", "right_paren"),
    ("*", "astrisk"),
    ("+", "plus"),
    (",", "comma"),
    ("-", "dash"),
    (".", "period"),
    ("/", "slash"),
    (":", "colon"),
    (";", "semicolon"),
    ("<", "less_than"),
    ("=", "equal"),
    (">", "greater_than"),
    ("?", "question_mark"),
    ("@", "at"),
    ("[", "left_bracket"),
    ("\\", "backslash"),
    ("]", "right_bracket"),
    ("^", "carrot"),
    ("_", "underscore"),
    ("`", "backtick"),
    ("|", "bar"),
    ("}", "right_brace"),
    ("{", "left_brace"),
    ("~", "tilde"),
    (" ", "space"),
    ("\t", "tab")
]

disallowed_list = [char for char, _ in disallowed]

class PRC(): #PregexContinuation
    def __init__(self, f, arity=0, args=[]):
        self.f = f  
        self.arity = arity
        self.args = args

    def __call__(self, pre):

        if self.arity == len(self.args):
            if self.arity == 0: return pregex.Concat([self.f, pre]) 
            elif self.arity == 1: return pregex.Concat([self.f(*self.args), pre])
            else: return pregex.Concat([self.f(self.args), pre]) #this line is bad, need brackets around input to f if f is Alt
        else: return PRC(self.f, self.arity, args=self.args+[pre(pregex.String(""))])


def concatPrimitives():
    return [Primitive("string_" + i, arrow(tpregex, tpregex), PRC(pregex.String(i))) for i in printable[:-4] if i not in disallowed_list
            ] + [
        Primitive("string_" + name, arrow(tpregex, tpregex), PRC(pregex.String(char))) for char, name in disallowed
    ] + [
        Primitive("r_dot", arrow(tpregex, tpregex), PRC(pregex.dot)),
        Primitive("r_d", arrow(tpregex, tpregex), PRC(pregex.d)),
        Primitive("r_s", arrow(tpregex, tpregex), PRC(pregex.s)),
        Primitive("r_w", arrow(tpregex, tpregex), PRC(pregex.w)),
        Primitive("r_l", arrow(tpregex, tpregex), PRC(pregex.l)),
        Primitive("r_u", arrow(tpregex, tpregex), PRC(pregex.u)),
        #todo
        Primitive("r_kleene", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.KleeneStar,1)),
        Primitive("r_plus", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Plus,1)),
        Primitive("r_maybe", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Maybe,1)),
        Primitive("r_alt", arrow(arrow(tpregex, tpregex) , arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Alt, 2)),
    ]

def strConstConcatPrimitives():
    return [Primitive("string_" + i, arrow(tpregex, tpregex), PRC(pregex.String(i))) for i in printable[:-4] if i not in disallowed_list
            ] + [
        Primitive("string_" + name, arrow(tpregex, tpregex), PRC(pregex.String(char))) for char, name in disallowed
    ] + [
        Primitive("r_dot", arrow(tpregex, tpregex), PRC(pregex.dot)),
        Primitive("r_d", arrow(tpregex, tpregex), PRC(pregex.d)),
        Primitive("r_s", arrow(tpregex, tpregex), PRC(pregex.s)),
        Primitive("r_w", arrow(tpregex, tpregex), PRC(pregex.w)),
        Primitive("r_l", arrow(tpregex, tpregex), PRC(pregex.l)),
        Primitive("r_u", arrow(tpregex, tpregex), PRC(pregex.u)),
        #todo
        Primitive("r_kleene", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.KleeneStar,1)),
        Primitive("r_plus", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Plus,1)),
        Primitive("r_maybe", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Maybe,1)),
        Primitive("r_alt", arrow(arrow(tpregex, tpregex) , arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Alt, 2)),
    ] + [
    Primitive("r_const", arrow(tpregex, tpregex), None)
    ]


def reducedConcatPrimitives():
    #uses strConcat!!
    #[Primitive("empty_string", arrow(tpregex, tpregex), PRC(pregex.String("")))
            #] + [
    return [Primitive("string_" + i, arrow(tpregex, tpregex), PRC(pregex.String(i))) for i in printable[:-4] if i not in disallowed_list
            ] + [
        Primitive("string_" + name, arrow(tpregex, tpregex), PRC(pregex.String(char))) for char, name in disallowed
        ] + [
        Primitive("r_dot", arrow(tpregex, tpregex), PRC(pregex.dot)),
        Primitive("r_d", arrow(tpregex, tpregex), PRC(pregex.d)),
        Primitive("r_s", arrow(tpregex, tpregex), PRC(pregex.s)),
        #Primitive("r_w", arrow(tpregex, tpregex), PRC(pregex.w)),
        Primitive("r_l", arrow(tpregex, tpregex), PRC(pregex.l)),
        Primitive("r_u", arrow(tpregex, tpregex), PRC(pregex.u)),
        #todo
        Primitive("r_kleene", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.KleeneStar,1)),
        #Primitive("r_plus", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Plus,1)),
        Primitive("r_maybe", arrow(arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Maybe,1)),
        Primitive("r_alt", arrow(arrow(tpregex, tpregex) , arrow(tpregex, tpregex), arrow(tpregex,tpregex)), PRC(pregex.Alt, 2)),
    ] + [
    Primitive("r_const", arrow(tpregex, tpregex), None)
    ]


def sketchPrimitives():
    return [Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
            ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, pregex.dot),
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene_5),
        Primitive("r_plus", arrow(tpregex, tpregex), _plus_5),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]

def basePrimitives():
    return [Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
            ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, pregex.dot),
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]



def altPrimitives():
    return [
        Primitive("empty_string", tpregex, pregex.String(""))
    ] + [
        Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
    ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, pregex.dot),
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        #Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]

def alt2Primitives():
    return [
        Primitive("empty_string", tpregex, pregex.String(""))
    ] + [
        Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
    ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, pregex.dot),
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        #Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        #Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]

def easyWordsPrimitives():
    return [
        Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[10:62] if i not in disallowed_list
    ] + [
        Primitive("r_d", tpregex, pregex.d),
        Primitive("r_s", tpregex, pregex.s),
        #Primitive("r_w", tpregex, pregex.w),
        Primitive("r_l", tpregex, pregex.l),
        Primitive("r_u", tpregex, pregex.u),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]


#def _wrapper(x): return lambda y: y

#specials = [".","*","+","?","|"]
"""
>>> import pregex as pre
>>> abc = pre.CharacterClass("abc", [0.1, 0.1, 0.8], name="MyConcept")
>>> abc.sample()
'b'
>>> abc.sample()
'c'
>>> abc.sample()
'c'
>>> abc.match("c")
-0.2231435513142097
>>> abc.match("a")
-2.3025850929940455
>>> abc
MyConcept
>>> x = pre.KleeneStar(abc)
>>> x.match("aabbac")
-16.58809928020405
>>> x.sample()
''
>>> x.sample()
''
>>> x.sample()
'cbcacc'
>>> x
(KleeneStar 0.5 MyConcept)
>>> str(x)
'MyConcept*'
"""


def emp_dot(corpus): return pregex.CharacterClass(printable[:-4], emp_distro_from_corpus(corpus, printable[:-4]), name=".")

def emp_d(corpus): return pregex.CharacterClass(printable[:10], emp_distro_from_corpus(corpus, printable[:10]), name="\\d")

#emp_s = pre.CharacterClass(slist, [], name="emp\\s") #may want to forgo this one. 

def emp_dot_no_letter(corpus): return pregex.CharacterClass(printable[:10]+printable[62:], emp_distro_from_corpus(corpus, printable[:10]+printable[62:]), name=".")

def emp_w(corpus): return pregex.CharacterClass(printable[:62], emp_distro_from_corpus(corpus, printable[:62]), name="\\w")

def emp_l(corpus): return pregex.CharacterClass(printable[10:36], emp_distro_from_corpus(corpus, printable[10:36]), name="\\l")

def emp_u(corpus): return pregex.CharacterClass(printable[36:62], emp_distro_from_corpus(corpus, printable[36:62]), name="\\u")


def emp_distro_from_corpus(corpus, char_list):
    from collections import Counter
    c = Counter(char for task in corpus for example in task.examples for string in example[1] for char in string)
    n = sum(c[char] for char in char_list)
    return [c[char]/n for char in char_list]



def matchEmpericalPrimitives(corpus):
    return lambda: [
        Primitive("empty_string", tpregex, pregex.String(""))
    ] + [
        Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list
    ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, emp_dot(corpus) ),
        Primitive("r_d", tpregex, emp_d(corpus) ),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_w", tpregex, emp_w(corpus) ),
        Primitive("r_l", tpregex, emp_l(corpus) ),
        Primitive("r_u", tpregex, emp_u(corpus) ),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        #Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]

def matchEmpericalNoLetterPrimitives(corpus):
    return lambda: [
        Primitive("empty_string", tpregex, pregex.String(""))
    ] + [
        Primitive("string_" + i, tpregex, pregex.String(i)) for i in printable[:-4] if i not in disallowed_list + list(printable[10:62])
    ] + [
        Primitive("string_" + name, tpregex, pregex.String(char)) for char, name in disallowed
    ] + [
        Primitive("r_dot", tpregex, emp_dot_no_letter(corpus) ),
        Primitive("r_d", tpregex, emp_d(corpus) ),
        Primitive("r_s", tpregex, pregex.s),
        Primitive("r_kleene", arrow(tpregex, tpregex), _kleene),
        #Primitive("r_plus", arrow(tpregex, tpregex), _plus),
        #Primitive("r_maybe", arrow(tpregex, tpregex), _maybe),
        Primitive("r_alt", arrow(tpregex, tpregex, tpregex), _alt),
        Primitive("r_concat", arrow(tpregex, tpregex, tpregex), _concat),
    ]


if __name__=='__main__':
    concatPrimitives()
    from dreamcoder.program import Program

    p=Program.parse("(lambda (r_kleene (lambda (r_maybe (lambda (string_x $0)) $0)) $0))")
    print(p)
    print(p.runWithArguments([pregex.String("")]))

    prims = concatPrimitives()
    g = Grammar.uniform(prims)

    for i in range(100):
        prog = g.sample(arrow(tpregex,tpregex))
        preg = prog.runWithArguments([pregex.String("")])
        print("preg:", preg.__repr__())
        print("sample:", preg.sample())



