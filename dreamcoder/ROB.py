import pickle
import numpy as np

# from API import *

# from pointerNetwork import *
# from programGraph import *
import random
import string
from string import printable
import re
import dreamcoder.pregex as pre


"""
IMPLEMENTATION OF THE LANGUAGE OF THE ROBUST FILL THING
ALSO INPUT / OUTPUT GENERATION
ALSO PROGRAM SAMPLING

THIS THING IS CRAZY AS FUCKKkKkkKKKkKk good luck y'all


Overall Design Choice : 

    all parts of the grammar is defiend as a class with class method "generate"
    which samples a random expression from that production node forward. perhaps
    the best view point is that a class is the non-terminal and has the capability of
    constructing sub-trees all the way down to the terminal level and the resulting
    tree can be evaluated
"""

from dreamcoder.program import Application, Primitive, Index, Abstraction


from dreamcoder.ROBUT import _INDEX,_POSITION_K,_CHARACTER,_DELIMITER,_BOUNDARY,N_EXPRS,_POSSIBLE_TYPES,_POSSIBLE_DELIMS,_POSSIBLE_R, MAX_STR_LEN

from dreamcoder.ROBUT import RepeatAgent, get_rollout, ALL_BUTTS, RobState, apply_fs

import dreamcoder.ROBUT as BUTT

MAX_LEN = 5

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

disallowed = dict(disallowed)

def allowed(x):
    return disallowed.get(x, x)

def prodLookup(name):
    return Primitive.GLOBALS[name]

class P:
    
    """
    a program is a concat of multiple expressions
    """

    @staticmethod
    def generate():
        n_expr = random.randint(1, N_EXPRS)    
        return P([E.generate() for _ in range(n_expr)])

    def __init__(self, exprs):
        self.exprs = exprs
        self.constr = None
        for e in self.exprs:
            self.constr = add_constr(e.constr, self.constr)

    def str_execute(self, input_str):
        return "".join(e.str_execute(input_str) for e in self.exprs)

    def flatten(self):
        buttons = []
        for e in self.exprs:
            buttons.extend( e.flatten() + [BUTT.Commit()] )
        return buttons

    def __str__(self):
        return " | ".join( str(e) for e in self.exprs )

    def ecProg(self):
        progList = []
        for expr in self.exprs:
            #conststrs must be treated differently 
            if isinstance(expr.ee, ConstStr):
                progList.extend(expr.ecProg())
            else: progList.append(expr.ecProg())

        #print("proglist", progList)
        p = Index(0)
        for frag in list(reversed(progList)):
            try: p = Application(frag, p)
            except:
                import pdb; pdb.set_trace()

        #print(Abstraction(p))
        return Abstraction(p)

class E:
    """
    an expression :D
    F | N | N1(N2) | N(F) | ConstStr(c)
    """
    @staticmethod
    def generate():
        ee_choices = [
        lambda: F.generate(),
        lambda: N.generate(),
        lambda: Compose(N.generate(), N.generate()),
        lambda: Compose(N.generate(), F.generate()),
        lambda: ConstStr.generate(),
        ]
        return E(random.choice(ee_choices)())

    def __init__(self, ee):
        self.ee = ee
        self.constr = ee.constr

    def __str__(self):
        return str(self.ee)

    def str_execute(self, input_str):
        return self.ee.str_execute(input_str)

    def flatten(self):
        return self.ee.flatten()

    def ecProg(self):
        return self.ee.ecProg()

class Compose:
    """
    chain 2 things together :3
    """

    def __init__(self, f1, f2):
        self.f1, self.f2 = f1, f2
        self.constr = add_constr(f1.constr, f2.constr)

    def str_execute(self, input_str):
        return self.f1.str_execute(self.f2.str_execute(input_str))

    def flatten(self):
        return self.f2.flatten() + self.f1.flatten()

    def __str__(self):
        return f"{str(self.f1)}( {str(self.f2)} )"

    def ecProg(self):
        
        p2 = Abstraction(Application(self.f2.ecProg(), Index(0)))
        p1Original = self.f1.ecProg()
        f, args = p1Original.applicationParse()
        f1Name = f.name

        p1f = prodLookup(f1Name+'_n')

        p1 = Application(p1f, p2)
        for arg in args:
            p1 = Application(p1, arg)

        return p1


class F:
    """
    SubString | GetSpan
    """
    @staticmethod
    def generate():
        ee_choices = [
        lambda: SubString.generate(),
        lambda: GetSpan.generate(),
        ]
        return F(random.choice(ee_choices)())

    def __init__(self, ee):
        self.ee = ee
        self.constr = ee.constr

    def str_execute(self, input_str):
        return self.ee.str_execute(input_str)

    def flatten(self):
        return self.ee.flatten()

    def __str__(self):
        return str(self.ee)

    def ecProg(self):
        return self.ee.ecProg()

class SubString:
    """
    take substring from position k1 to k2
    """
    @staticmethod
    def generate():
        k1 = random.choice(_POSITION_K)
        k2 = random.choice(_POSITION_K)
        if k1 > k2:
            k1, k2 = k2, k1
        return SubString(k1, k2)

    def __init__(self, k1, k2):
        self.k1, self.k2 = k1, k2
        self.constr = ({}, k2)

    def str_execute(self, input_str):
        return input_str[self.k1:self.k2]

    def flatten(self):
        return [BUTT.SubStr1(self.k1), BUTT.SubStr2(self.k2)]

    def __str__(self):
        return "SubStr" + str((self.k1, self.k2))

    def ecProg(self):
        f = prodLookup("SubStr")
        a = prodLookup(f"pos_{self.k1}")
        b = prodLookup(f"pos_{self.k2}")
        return Application( Application(f, a), b)


class GetSpan:
    @staticmethod
    def generate():
        r1 = R.generate()
        r2 = R.generate()
        i1 = random.choice(_INDEX)
        i2 = random.choice(_INDEX)
        b1 = random.choice(_BOUNDARY)
        b2 = random.choice(_BOUNDARY)
        return GetSpan(r1, i1, b1, r2, i2, b2)

    def __init__(self, r1, i1, b1, r2, i2, b2):
        self.r1, self.i1, self.b1 = r1, i1, b1
        self.r2, self.i2, self.b2 = r2, i2, b2

        dic = {r1 : stepped_abs(i1), 
               r2 : stepped_abs(i2), 
              }
        self.constr = dic, 0

    def __str__(self):
        return "GetSpan"+str((self.r1, self.i1, self.b1, self.r2, self.i2, self.b2))

    def str_execute(self, input_str):
        """
        all complaints of this function please send to mnye@mit.edu
        evanthebouncy took no part in this :v
        """
        return input_str[[m.end() for m in re.finditer(self.r1[0], input_str)][self.i1] if self.b1 == "End" else [m.start() for m in re.finditer(self.r1[0], input_str)][self.i1] : [m.end() for m in re.finditer(self.r2[0], input_str)][self.i2] if self.b2 == "End" else [m.start() for m in re.finditer(self.r2[0], input_str)][self.i2]]


    def flatten(self):
        return [BUTT.GetSpan1(self.r1.name), BUTT.GetSpan2(self.i1), BUTT.GetSpan3(self.b1),
                BUTT.GetSpan4(self.r2.name), BUTT.GetSpan5(self.i2), BUTT.GetSpan6(self.b2)] 

    def ecProg(self):
        f = prodLookup("GetSpan")
        r1 = prodLookup(f"regex_{allowed(self.r1.name)}")
        i1 = prodLookup(f"index_{self.i1}")
        y1 = prodLookup(f"bound_{self.b1}")
        r2 = prodLookup(f"regex_{allowed(self.r2.name)}")
        i2 = prodLookup(f"index_{self.i2}")
        y2 = prodLookup(f"bound_{self.b2}")
        
        args = [r1, i1, y1, r2, i2, y2]
        p = f
        for arg in args:
            p = Application(p, arg)
        return p

class ConstStr:
    @staticmethod
    def generate():
        l = random.choice(list(range(1, MAX_LEN)))
        c = pre.create("."*l).sample()
        return ConstStr(c)

    def __init__(self, c):
        self.c = c
        self.constr = {}, 0

    def str_execute(self, input_str):
        raise NotImplementedError

    def flatten(self):

        r  = [BUTT.Const(self.c[0])]
        for const in self.c[1:]:
            r.append( BUTT.Commit() )
            r.append( BUTT.Const(const) )

        return r

    def __str__(self):
        return "ConstStr "+str((self.c))

    def ecProg(self):
        cs = []
        for c in self.c: #char-wise
            name = f"char_{allowed(c)}"
            cs.append(prodLookup(name))
        return cs

class N:
    @staticmethod
    def generate():
        choices = [
            GetToken,
            ToCase,
            Replace,
            GetUpTo,
            GetFrom,
            GetFirst,
            GetAll ]
        return random.choice(choices).generate()

    #def __init__(self, name):
    # TODO

class GetToken:
    @staticmethod
    def generate():
        t = R.generate_type()
        i = random.choice(_INDEX)
        return GetToken(t, i)

    def __init__(self, t, i):
        self.t, self.i = t, i

        dic = {t : stepped_abs(i), 
              }
        self.constr = dic, 0

    def __str__(self):
        return "GetToken" + str((self.t, self.i))

    def str_execute(self, input_str):
        return re.findall(self.t[0], input_str)[self.i]

    def flatten(self):
        return [BUTT.GetToken1(self.t.name), BUTT.GetToken2(self.i)] 

    def ecProg(self):
        f = prodLookup("GetToken")
        t = prodLookup(f"type_{self.t.name}")
        i = prodLookup(f"index_{self.i}")
        return Application(Application(f, t), i)

class ToCase:

    candidates = [
        ("Proper", lambda x : x.title()),
        ("AllCaps", lambda x: x.upper()),
        ("Lower", lambda x: x.lower()),
        ]
    @staticmethod
    def generate():
        return ToCase(random.choice(ToCase.candidates))
    def __init__(self, ss):
        self.name, self.s = ss
        dic = { R("Alphanum"): 1 }
        self.constr = dic, 0
        #todo

    def flatten(self):
        return [ BUTT.ToCase(self.name) ]

    def str_execute(self, input_str):
        raise NotImplementedError

    def __str__(self):
        return "ToCase"+self.name

    def ecProg(self):
        return prodLookup("ToCase_"+self.name)

class Replace:

    @staticmethod
    def generate():
        d1 = random.choice(_DELIMITER)
        d2 = random.choice([d for d in _DELIMITER if d != d1])
        return Replace(d1, d2)

    def __init__(self, d1, d2):
        self.d1 = d1
        self.d2 = d2
        dic = {R(d1): 1}
        self.constr = dic, 0

    def __str__(self):
        return "Replace"+str((self.d1, self.d2))

    def flatten(self):
        return [ BUTT.Replace1(self.d1), BUTT.Replace2(self.d2) ] 

    def str_execute(self, input_str):
        raise NotImplementedError

    def ecProg(self):
        f = prodLookup("Replace")
        d1 = prodLookup("delim_" + allowed(self.d1))
        d2 = prodLookup("delim_" + allowed(self.d2))
        return Application(Application(f, d1), d2)

class Trim:
    pass

class GetUpTo:
    @staticmethod
    def generate():
        r = R.generate()
        #i = random.choice(_INDEX)
        return GetUpTo(r)

    def __init__(self, r):
        self.r = r

        dic = {r : 1, 
              }
        self.constr = dic, 0

    def __str__(self):
        return "GetUpTo" + str(self.r)

    def str_execute(self, input_str):
        raise NotImplementedError

    def flatten(self):
        return [ BUTT.GetUpTo(self.r.name) ]

    def ecProg(self):
        f = prodLookup("GetUpTo")
        r = prodLookup("regex_"+allowed(self.r.name))
        return Application(f, r)

class GetFrom:
    @staticmethod
    def generate():
        r = R.generate()
        #i = random.choice(_INDEX)
        return GetFrom(r)

    def __init__(self, r):
        self.r = r

        dic = {r : 1, 
              }
        self.constr = dic, 0

    def __str__(self):
        return "GetFrom" + str(self.r)

    def str_execute(self, input_str):
        raise NotImplementedError

    def flatten(self):
        return [ BUTT.GetFrom(self.r.name)] 

    def ecProg(self):
        f = prodLookup("GetFrom")
        r = prodLookup("regex_"+allowed(self.r.name))
        return Application(f, r)    

class GetFirst:
    @staticmethod
    def generate():
        t = R.generate_type()
        i = random.choice(_INDEX)
        return GetFirst(t, i)

    def __init__(self, t, i):
        self.t, self.i = t, i

        dic = {t : stepped_abs(i), 
              }
        self.constr = dic, 0

    def __str__(self):
        return "GetFirst" + str((self.t, self.i))

    def str_execute(self, input_str):
        raise NotImplementedError

    def flatten(self):
        return [ BUTT.GetFirst1(self.t.name), BUTT.GetFirst2( self.i ) ]

    def ecProg(self):
        f = prodLookup("GetFirst")
        t = prodLookup(f"type_{self.t.name}")
        i = prodLookup(f"index_{self.i}")
        return Application(Application(f, t), i)

class GetAll:
    @staticmethod
    def generate():
        t = R.generate_type()
        #i = random.choice(_INDEX)
        return GetAll(t)

    def __init__(self, t):
        self.t = t

        dic = {t : 1, 
              }
        self.constr = dic, 0

    def __str__(self):
        return "GetAll" + str((self.t))

    def str_execute(self, input_str):
        raise NotImplementedError

    def flatten(self):
        return [ BUTT.GetAll(self.t.name)]

    def ecProg(self):
        f = prodLookup("GetAll")
        t = prodLookup("type_"+allowed(self.t.name))
        return Application(f, t)

class R:

    @staticmethod
    def generate_type():
        type_choice = random.choice(list(_POSSIBLE_TYPES.keys()))
        return R(type_choice)

    @staticmethod
    def generate_delim():
        
        delim_choice = random.choice(list(_POSSIBLE_DELIMS.keys()))
        return R(delim_choice)


    @staticmethod
    def generate():
        if np.random.random() < 0.5:
            return R.generate_type()
        else:
            return R.generate_delim()

    def __init__(self, name):
        self.name = name
        regex = _POSSIBLE_R[name]
        self.ree, self.pre = regex

    def __getitem__(self, key):
        if key == 0 : 
            return self.ree
        if key == 1 :
            return self.pre
        assert 0, "you ve gone too far"

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return self.__str__()

    def str_execute(self, input_str):
        assert 0, "what u doing boi"


def generate_string(constraint, max_string_size=MAX_STR_LEN):
    constraint_dict, min_size = constraint
    #sample a size from min to max
    size = random.randint(min_size, max_string_size)
    indices = set(range(size))
    slist = random.choices(printable[:-4] , k=size)
    # schematically:
    #print("min_size", min_size)
    #print("size", size)
    for item in constraint_dict:
        reg, preg = item.ree, item.pre
        num_to_insert = max(0, constraint_dict[item] - len(re.findall(reg, ''.join(slist))))
        if len(indices) < num_to_insert: return None
        indices_to_insert = set(random.sample(indices, k=num_to_insert))
      
        for i in indices_to_insert:
            slist[i] = preg.sample()
        indices = indices - indices_to_insert
    string = ''.join(slist)
    if len(string) > max_string_size: return string[:max_string_size] 
    return string

def executeProg(prog, inp):
    return BUTT.apply_fs(BUTT.RobState.new([inp], [""]), prog.flatten()).committed[0]

def generate_FIO(n_ios, verbose=False):
    """
        generate a function, inputs, outputs triple
    """
    prog = P.generate()
    inputs = []
    outputs = []
    for _ in range(20):
        if len(inputs) == n_ios:
            return prog, inputs, outputs
        try:
            inp = generate_string(prog.constr)
            out = BUTT.apply_fs(BUTT.RobState.new([inp], [""]), prog.flatten()).committed[0]
            # make sure the outputs are well conditioned
            if len(out) > max(_POSITION_K):
                continue
            if len(out) == 0:
                continue
            inputs.append(inp)
            outputs.append(out)
        except Exception as e:
            #print (e)
            pass

    if verbose: print ("gneration failed retrying")
    return generate_FIO(n_ios)

def get_supervised_sample(n_ios=4,
                          render_kind={'render_scratch' : 'yes',
                                       'render_past_buttons' : 'no'}):
    
    prog, inputs, outputs = generate_FIO(n_ios)
    env = BUTT.ROBENV(inputs, outputs, render_kind)
    repeat_agent = RepeatAgent(prog.flatten())
    trace = get_rollout(env, repeat_agent, 30)

    states = [x[0] for x in trace]
    actions = [x[1] for x in trace]
    return states, actions



################################### UTILS #################################
# merge 2 constraint dictionaries together
# a constraint dictionary keep track of how many tokens are to be needed on input
def add_constr(c1, c2=None):
    if c2 is None:
        return c1
    d1, m1 = c1
    d2, m2 = c2
    min_size = max(m1, m2)
    d = {}
    for item in set(d1.keys()) | set(d2.keys()):
        d[item] = max(d1.get(item, 0), d2.get(item,0))
    return d, min_size

def stepped_abs(x):
    return x + 1 if x >= 0 else abs(x)

# =================== TESTS ========================

def test1():
    pstate = RobState.new(["12A", "2A4", "A45", "4&6", "&67"],
                          ["", "", "", "", ""])
    print (ALL_BUTTS)
    print (len(ALL_BUTTS))
    print (pstate)
    fs = [
            BUTT.ToCase("Lower"),
            BUTT.Replace1("&"),
            BUTT.Replace2("["),
            BUTT.SubStr1(1),
            BUTT.SubStr2(2),
            BUTT.Commit(),
            ]

    print (apply_fs(pstate, fs))

def test2():
    print (ALL_BUTTS)
    print (len(ALL_BUTTS))
    pstate = RobState.new(["Mr.Pu", "Mr.Poo"],
                          ["", ""])
    fs = [
            BUTT.GetToken1("Word"),
            BUTT.GetToken2(1),
            BUTT.Commit(),
            ]
    print (apply_fs(pstate, fs))
    
    gs = [
            BUTT.GetUpTo("."),
            BUTT.Commit(),
            ]
    print (apply_fs(pstate, gs))

    hs = [
            BUTT.GetFrom("."),
            BUTT.Commit(),
            ]
    print (apply_fs(pstate, hs))

    ts = [
            BUTT.GetFirst1("Word"),
            BUTT.GetFirst2(2),
            BUTT.Commit(),
            ]
    print (apply_fs(pstate, ts))

    vs = [
            BUTT.GetAll("Word"),
            BUTT.Commit(),
            ]
    print (apply_fs(pstate, vs))

def test3():
    print (ALL_BUTTS)
    print (len(ALL_BUTTS))
    pstate = RobState.new(["(hello)123", "(mister)123"],
                          ["HELLORE", "MISTERRE"])
    fs = [
            BUTT.GetSpan1("("),
            BUTT.GetSpan2(0),
            BUTT.GetSpan3("End"),
            BUTT.GetSpan4(")"),
            BUTT.GetSpan5(0),
            BUTT.GetSpan6("Start"),
            BUTT.ToCase("AllCaps"),
            BUTT.Commit(),
            BUTT.Const("R"),
            BUTT.Commit(),
            BUTT.Const("E"),
            BUTT.Commit(),
            ]
    print (apply_fs(pstate, fs))

def test4():
    pstate = RobState.new(["(hello)1)23", "(mis)ter)123"],
                          ["HELLO", "MIS"])
    fs = [
            BUTT.GetSpan1("("), 
            BUTT.GetSpan2(0), 
            BUTT.GetSpan3("End"), 
            BUTT.GetSpan4(")"), 
            BUTT.GetSpan5(0), 
            BUTT.GetSpan6("Start"), 
            BUTT.ToCase("AllCaps"),
            BUTT.Commit(),
         ]
    pstate_new = apply_fs(pstate, fs)
    _, scratch, _, _, masks, _ = pstate_new.to_np()

def test5():
    pstate = RobState.new(["123hello123goodbye1234hola123231"],
                          ["dontreadthis"])
    fs = [
            BUTT.GetToken1("Word"),
         ]
    pstate_new = apply_fs(pstate, fs)
    print (pstate_new)
    _, scratch, _, _, masks, _ = pstate_new.to_np()
    print (scratch[0])
    print (masks[0])

def test6():
    pstate = RobState.new(["123hello123goodbye1234hola123231"],
                          ["dontreadthis"])
    fs = [
            BUTT.GetSpan1("Word"),
            BUTT.GetSpan2(1),
            BUTT.GetSpan3("End"),
            BUTT.GetSpan4("Number"),
            BUTT.GetSpan5(3),
         ]
    pstate_new = apply_fs(pstate, fs)
    _, scratch, _, _, masks, _ = pstate_new.to_np()
    print (scratch[0])
    print (masks[0])

def test7():

    prog, inputs, outputs = generate_FIO(5)
    env = BUTT.ROBENV(inputs, outputs)
    repeat_agent = RepeatAgent(prog.flatten())
    trace = get_rollout(env, repeat_agent, 30)
    print ([(x[1],x[2]) for x in trace])

def test8():
    S, A = BUTT.get_supervised_sample()
    print ("generated these number of states", len(S))
    print ("generated these number of actions", len(A))

    print ("============ first state")
    inputs, scratch, committed, outputs, masks, last_butt = S[0]
    print ("shapes of inputs, scratch, committed, outputs")
    print (inputs.shape)
    print (scratch.shape)
    print (committed.shape)
    print (outputs.shape)
    print ("shape of mask")
    print (masks.shape)
    print ("last_butt is just a number")
    print (last_butt)
    print ("first action")
    print (A[0])

    print ("============ second state")
    inputs, scratch, committed, outputs, masks, last_butt = S[1]
    print ("shapes of inputs, scratch, committed, outputs")
    print (inputs.shape)
    print (scratch.shape)
    print (committed.shape)
    print (outputs.shape)
    print ("shape of mask")
    print (masks.shape)
    print ("last_butt is just a number")
    print (last_butt)
    print ("second action")
    print (A[1])

def test9():
    S, A = BUTT.get_supervised_sample()
    print ("generated these number of states", len(S))
    inputs, scratch, committed, outputs, masks, last_butt = S[0]
    print ("shapes of inputs, scratch, committed, outputs")
    print (inputs.shape)
    print (scratch.shape)
    print (committed.shape)
    print (outputs.shape)
    print ("shape of mask")
    print (masks.shape)
    print ("last_butt is just a number")
    print (last_butt)
    from robut_net import Agent
    agent = Agent(ALL_BUTTS)
    chars, masks, last_butts = agent.states_to_tensors(S)
    print("chars shape")
    print(chars.shape)
    print("masks shape")
    print(masks.shape)
    print("last_butts shape")
    print(last_butts.shape)
    print(last_butts)

    num_params = sum(p.numel() for p in agent.nn.parameters() if p.requires_grad)
    print("num params:", num_params)

    for i in range(200):
        loss = agent.learn_supervised(S,A)
        if i%10 == 0: print(i, loss)
    j = 4
    char, mask, last_butt = agent.states_to_tensors(S)#[S[j]])
    dist = agent.nn.forward(char, mask, last_butt)
    _, argmax = dist.max('actions')

    print("real action", agent.idx[A[j].name])
    print("selected_action", argmax)

def test10():
    S, A = BUTT.get_supervised_sample()
    # print ("generated these number of states", len(S))
    from robut_net import Agent
    agent = Agent(ALL_BUTTS)
    for i in range(400):
        loss = agent.learn_supervised(S,A)
        if i%10 == 0: print(f"iteration {i}, loss: {loss.item()}")
    actions = agent.best_actions(S)
    print("real actions:")
    print(A)
    print("model actions:")
    print(actions)

def test11():

    for i in range(1000):

        prog, inputs, outputs = generate_FIO(5)
        env = BUTT.ROBENV(inputs, outputs)
        env.verbose = True
        repeat_agent = RepeatAgent(prog.flatten())
        drop_idx = random.choice(range(len(repeat_agent.btns)-1))
        repeat_agent.btns = repeat_agent.btns[:drop_idx-1] + repeat_agent.btns[drop_idx:]
        trace = get_rollout(env, repeat_agent, 30)
    #    print ([x[1:3] for x in trace])

def test12():
    '''
    get the statistics of all the buttons
    '''
    ALL_A = dict()
    for i in range(100000):
        S, A = BUTT.get_supervised_sample()
        ob_list = [str(s) for s in S]
            
        for a in A:
            if a.name not in ALL_A:
                ALL_A[a.name] = 0
            ALL_A[a.name] += 1

        if i % 1000 == 0:
            xx = []
            for b in BUTT.ALL_BUTTS:
                b_name = b.name
                if b_name not in ALL_A:
                    xx.append(0)
                else:
                    xx.append(ALL_A[b_name])

            import matplotlib.pyplot as plt
            
            objects = [b.name for b in BUTT.ALL_BUTTS][:-1]
            y_pos = np.arange(len(objects))
            performance = xx[:-1]

            fig, ax = plt.subplots(figsize=(50, 100))

            ax.barh(y_pos, performance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(objects)

            plt.savefig('butt_distr.png')

def test13():
    for i in range(100000):
        print (i)
        env = BUTT.ROBENV(inputs, outputs)
        repeat_agent = BUTT.RepeatAgent(prog.flatten())
        trace = get_rollout(env, repeat_agent, 30)

        str_obs = [str(x[0]) for x in trace]
        if len(str_obs) != len(set(str_obs)):
            for i in range(len(trace)):
                for j in range(len(trace)):
                    if i != j:
                        if str(trace[i][0]) == str(trace[j][0]):
                            print ("collision across ")
                            print (i)
                            print (j)
                            import pdb; pdb.set_trace()

def test14():
    print (' ================ our thing : normal render')
    S, A = get_supervised_sample()
    print ('scratch\n', S[-2][1])
    print ('last btn type\n', S[-1][-2])
    print ('past btns\n', S[-1][-1])

    print (' ================ karel thing : render past buttons as well')
    S, A = get_supervised_sample(render_kind={'render_scratch' : 'yes',
                                              'render_past_buttons' : 'yes'})
    print ('scratch\n', S[-2][1])
    print ('last btn type\n', S[-1][-2])
    print ('past btns\n', S[-1][-1])

    print (' ================= robust fill thing : render past buttons but no scratch')
    S, A = get_supervised_sample(render_kind={'render_scratch' : 'no',
                                              'render_past_buttons' : 'yes'})
    print ('scratch\n', S[-2][1])
    print ('last btn type\n', S[-1][-2])
    print ('past btns\n', S[-1][-1])

def test15():
    pstate = RobState.new(["123hello123goodbye1234hola123231"],
                          ["dontreadthis"])
    fs = [
            BUTT.GetFrom("Char"),
            BUTT.Commit(),
         ]
    pstate_new = apply_fs(pstate, fs)
    _, scratch, _, _, masks, _ = pstate_new.to_np()
    print (scratch[0])
    print (masks[0])

def test16():
    prog, inputs, outputs = generate_FIO(5)
    env = BUTT.ROBENV(inputs, outputs, render_kind = 'ablate_scratch')
    repeat_agent = BUTT.RepeatAgent(prog.flatten())
    trace = get_rollout(env, repeat_agent, 30)
    obs = [x[0] for x in trace]
    print (obs[-1])

def test17():
    for i in range(100000):
        print (i)
        prog, inputs, outputs = generate_FIO(5)
        print(inputs)
        print(outputs)
        p = prog.flatten()
        print(p)
        print("len:", len(p))
        print('\n')
        env = BUTT.ROBENV(inputs, outputs)
        repeat_agent = BUTT.RepeatAgent(prog.flatten())
        trace = get_rollout(env, repeat_agent, 30)

        str_obs = [str(x[0]) for x in trace]
        if len(str_obs) != len(set(str_obs)):
            for i in range(len(trace)):
                for j in range(len(trace)):
                    if i != j:
                        if str(trace[i][0]) == str(trace[j][0]):
                            print ("collision across ")
                            print (i)
                            print (j)
                            import pdb; pdb.set_trace()

if __name__ == '__main__':
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # # test9() crashes
    # # test10() crashes
    # test11()
    # test12()
    # test13()
    # test14()
    # test16()
    test17()
