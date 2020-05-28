import pickle
import numpy as np

import random
import string
from string import printable
import re
import dreamcoder.pregex as pre

import traceback

_INDEX = list(range(-5, 6))

MAX_STR_LEN = 36

_POSITION_K = list(range(-MAX_STR_LEN, MAX_STR_LEN+1))
_CHARACTER = string.printable[:-4]
_DELIMITER = "& , . ? ! @ ( ) [ ] % { } / : ; $ # \" ' -".split(' ') + [" "]
_BOUNDARY = ["Start", "End"]

N_EXPRS = 6

_POSSIBLE_TYPES = {
        "Number" :   (r'[0-9]+', pre.create('\\d+')),
        "Word" :     (r'([A-z])+', pre.create('\\w+')),
        "Alphanum" : (r'[A-z]', pre.create('\\w')),
        "PropCase" : (r'[A-Z][a-z]+', pre.create('\\u\\l+')),
        "AllCaps" :  (r'[A-Z]', pre.create('\\u')),
        "Lower" :    (r'[a-z]', pre.create('\\l')),
        "Digit" :    (r'[0-9]', pre.create('\\d')),
        "Char" :     (r'.', pre.create('.')),
        }

_POSSIBLE_DELIMS = {}
for i in _DELIMITER:
    j = i
    if j in ['(', ')', '.']: 
        j = re.escape(j)
    _POSSIBLE_DELIMS[i] = (re.escape(i), pre.create(j))

_POSSIBLE_R = {**_POSSIBLE_TYPES, **_POSSIBLE_DELIMS}



N_IO = 4

class RobState:

    @staticmethod
    def crash_state_np(render_kind):
        return RobState.new(["" for _ in range(N_IO)], ["" for _ in range(N_IO)]).to_np(render_kind)

    @staticmethod
    def new(inputs, outputs):
        assert len(inputs) == len(outputs)
        committed = ["" for _ in range(len(inputs))]
        scratch = [x for x in inputs]
        past_buttons = []
        return RobState(inputs, scratch, committed, outputs, past_buttons)

    def __init__(self, inputs, scratch, committed, outputs, past_buttons):

        self.inputs    = [x for x in inputs]
        self.scratch   = [x for x in scratch]
        self.committed = [x for x in committed]
        self.outputs   = [x for x in outputs]
        self.past_buttons = [x for x in past_buttons]

    def copy(self):
        return RobState(self.inputs,
                        self.scratch,
                        self.committed,
                        self.outputs,
                        self.past_buttons)

    def __repr__(self):
        return str((self.inputs, self.scratch, self.committed, self.outputs, self.past_buttons))

    def __str__(self):
        return self.__repr__()

    def str_to_np(self, list_of_str):
        """
        turn a list of string into a np representation
        can be made faster with a dict, i think ...
        """
        ret = np.zeros(shape = (len(list_of_str), max(_POSITION_K)))
        for i, strr in enumerate(list_of_str):
            for j, char in enumerate(strr):
                ret[i][j] = _CHARACTER.index(char) + 1
        return ret

    def get_list_btns(self):
        if len(self.past_buttons) == 0:
            return np.array([0])
        else:
            # return np.array([ALL_BUTTS.index(btn) + 1 for btn in self.past_buttons])
            return np.array([ALL_BUTTS_NAME_MAP[btn.name] + 1 for btn in self.past_buttons])

    def get_last_btn_type(self):
        last_butt_type = 0 if len(self.past_buttons) == 0 else ALL_BUTTS_TYPES.index(self.past_buttons[-1].__class__) + 1
        return last_butt_type
        
    def to_np(self, render_kind):
        render_scratch = render_kind['render_scratch']
        render_past_buttons = render_kind['render_past_buttons']
        assert render_scratch in ['yes', 'no']
        assert render_past_buttons in ['yes', 'no']

        # create all the useful informations and mask them out as needed
        if self.past_buttons == []:
            masks = [Button.str_masks_to_np_default() for _ in range(len(self.inputs))]
        else:
            masks = [self.past_buttons[-1].str_masks_to_np(str1, self) for str1 in self.scratch]

        rendered_inputs = self.str_to_np(self.inputs)
        rendered_scratch = self.str_to_np(self.scratch)
        rendered_committed = self.str_to_np(self.committed)
        rendered_outputs = self.str_to_np(self.outputs)
        rendered_masks = np.array(masks)
        rendered_last_butt_type = self.get_last_btn_type()
        rendered_past_buttons = self.get_list_btns() 

        # start masking stuff off depend on parameters
        if render_scratch == 'no':
            rendered_scratch = np.zeros(shape = (len(self.scratch), max(_POSITION_K)))
            rendered_masks = [Button.str_masks_to_np_default() for _ in range(len(self.inputs))]
        if render_past_buttons == 'no':
            rendered_past_buttons = np.array([0])

        return (rendered_inputs,
                rendered_scratch,
                rendered_committed,
                rendered_outputs,
                rendered_masks,
                rendered_last_butt_type,
                rendered_past_buttons,
                )

# ===================== BUTTONS ======================

class Button:

    @staticmethod
    def str_masks_to_np_default():
        """
            len(_INDEX) number of regex masks
            and
            1 for replace
            1 for substring
        """
        np_masks = np.zeros(shape = (max(_INDEX) + 2, max(_POSITION_K)))
        return np_masks

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    # check if the next button is legal or not legal
    def check_next_btn(self, nxt_btn):
        return True

    def str_masks_to_np(self, str1, pstate):
        return Button.str_masks_to_np_default()

    def __eq__(self, other):
        return self.name == other.name


class Commit(Button):
    @staticmethod
    def generate_buttons():
        return [Commit()]

    def __init__(self):
        self.name = f"Commit"

    def __call__(self, pstate):
        scratch_new = pstate.inputs
        committed_new = [x[0]+x[1] for x in zip(pstate.committed, pstate.scratch)]
        # check we actually committed stuff
        check_change(pstate.committed,committed_new)
        # check commit is sensible
        for commit, output in zip(committed_new, pstate.outputs):
            if output == "":
                continue
            if not output.startswith(commit):
                raise CommitPrefixError
        return RobState(pstate.inputs,
                        scratch_new,
                        committed_new,
                        pstate.outputs,
                        pstate.past_buttons + [self])


class ToCase(Button):

    @staticmethod
    def generate_buttons():
        ss = ["Proper", "AllCaps", "Lower"]
        ret = [ToCase(s) for s in ss]
        return ret

    def __init__(self, s):
        self.name = f"ToCase({s})"
        self.s = s

    def f(self, x):
        if self.s == "Proper":
            return x.title()
        if self.s == "AllCaps":
            return x.upper()
        if self.s == "Lower":
            return x.lower()



    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class Replace1(Button):
    @staticmethod
    def generate_buttons():
        return [Replace1(d1) for d1 in _DELIMITER]

    def __init__(self, d1):
        self.name = f"Replace1({d1})"
        self.d1 = d1

    def check_next_btn(self, nxt_btn):
        if "Replace2" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        str_masks = Button.str_masks_to_np_default()
        idxs = [pos for pos, char in enumerate(str1) if char == self.d1]
        str_masks[-2][idxs] = 1
        return str_masks

    def __call__(self, pstate):
        ret =  RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])
        return ret

class Replace2(Button):
    @staticmethod
    def generate_buttons():
        return [Replace2(d2) for d2 in _DELIMITER]

    def __init__(self, d2):
        self.name = f"Replace2({d2})"
        self.d2 = d2

    def __call__(self, pstate):
        if "Replace1" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        d1 = pstate.past_buttons[-1].d1
        scratch_new = [x.replace(d1, self.d2) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])


class SubStr1(Button):

    @staticmethod
    def generate_buttons():
        ret = [SubStr1(k1) for k1 in _POSITION_K]
        return ret

    def __init__(self, k1):
        self.name = f"SubStr1({k1})"
        self.k1 = k1

    def check_next_btn(self, nxt_btn):
        if "SubStr2" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        str_masks = Button.str_masks_to_np_default()
        mask_sub = str_masks[-1]
        mask_sub[self.k1:] = 1
        return str_masks

    def __call__(self, pstate):
        ret =  RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])
        return ret

class SubStr2(Button):

    @staticmethod
    def generate_buttons():
        ret = [SubStr2(k2) for k2 in _POSITION_K]
        return ret

    def __init__(self, k2):
        self.name = f"SubStr2({k2})"
        self.k2 = k2

    def __call__(self, pstate):
        if "SubStr1" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        # get the k1 from the previous button
        k1 = pstate.past_buttons[-1].k1
        scratch_new = [x[k1:self.k2] for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetToken1(Button):

    @staticmethod
    def generate_buttons():
        return [GetToken1(name) for name in _POSSIBLE_TYPES.keys()] 

    def __init__(self, rname):
        self.name = f"GetToken1({rname})"
        self.rname = rname
        self.t = _POSSIBLE_TYPES[rname]

    def check_next_btn(self, nxt_btn):
        if "GetToken2" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        str_masks = Button.str_masks_to_np_default()
        # enumerate over all the regex masks
        p = list(re.finditer(self.t[0], str1))
        for i, m in enumerate(p[:max(_INDEX)]):
            str_masks[i][m.start():m.end()] = 1
        return str_masks

    def __call__(self, pstate):
        return  RobState(pstate.inputs,
                         pstate.scratch,
                         pstate.committed,
                         pstate.outputs,
                         pstate.past_buttons + [self])

class GetToken2(Button):

    @staticmethod
    def generate_buttons():
        return [GetToken2(i) for i in _INDEX] 

    def __init__(self, i):
        self.name = f"GetToken2({i})"
        self.i = i

    def f(self, x, t):
        # print (t[0])
        allz = re.finditer(t[0], x)
        match = list(allz)[self.i]
        return x[match.start():match.end()]

    def __call__(self, pstate):
        if "GetToken1" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        # get the type from GetToken1 button
        t = pstate.past_buttons[-1].t
        scratch_new = [self.f(x, t) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetUpTo(Button):

    @staticmethod
    def generate_buttons():
        return [GetUpTo(name) \
            for name in _POSSIBLE_R.keys()] 

    def __init__(self, rname):
        self.name = f"GetUpTo({rname})"
        self.rname = rname
        self.r = _POSSIBLE_R[rname]


    def f(self, string): 
        return string[:[m.end() \
            for m in re.finditer(self.r[0], string)][0]]

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetFrom(Button):

    @staticmethod
    def generate_buttons():
        return [GetFrom(name) \
            for name in _POSSIBLE_R.keys()] 

    def __init__(self, rname):
        self.name = f"GetFrom({rname})"
        self.rname = rname
        self.r = _POSSIBLE_R[rname] 


    def f(self, string): 
        return string[[m.end() \
            for m in re.finditer(self.r[0], string)][-1]:]

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetFirst1(Button):

    @staticmethod
    def generate_buttons():
        return [GetFirst1(name) for name in _POSSIBLE_TYPES.keys()]

    def __init__(self, rname):
        self.name = f"GetFirst1({rname})"
        self.rname = rname
        self.t = _POSSIBLE_TYPES[rname]

    def check_next_btn(self, nxt_btn):
        if "GetFirst2" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        str_masks = Button.str_masks_to_np_default()
        # enumerate over all the regex masks
        p = list(re.finditer(self.t[0], str1))
        for i, m in enumerate(p[:max(_INDEX)]):
            str_masks[i][m.start():m.end()] = 1
        return str_masks

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetFirst2(Button):

    @staticmethod
    def generate_buttons():
        return [GetFirst2(i) for i in _INDEX] 

    def __init__(self, i):
        self.name = f"GetFirst2({i})"
        self.i = i

    def f(self, string, t):
        xx = [string[x.start():x.end()] for x in list(re.finditer(t[0], string))]
        # print("xx", xx)
        # print("i", i)
        # print(type(i))
        return "".join(xx[:(self.i+1)])

    def __call__(self, pstate):
        if "GetFirst1" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        t = pstate.past_buttons[-1].t
        scratch_new = [self.f(x, t) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetAll(Button):

    @staticmethod
    def generate_buttons():
        return [GetAll(name) \
            for name in _POSSIBLE_TYPES.keys()] 

    def __init__(self, rname):
        self.name = f"GetAll({rname})"
        self.rname = rname
        self.r = _POSSIBLE_TYPES[rname]

    def f(self, string):
        xx = [string[x.start():x.end()] for x in list(re.finditer(self.r[0], string))]
        return "".join(xx)

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan1(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan1(name) for name in _POSSIBLE_R.keys()]

    def __init__(self, rname):
        self.name = f"GetSpan1({rname})"
        self.rname = rname
        self.r1 = _POSSIBLE_R[rname]

    def check_next_btn(self, nxt_btn):
        if "GetSpan2" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        return get_span_mask_render(str1, pstate.past_buttons[-1:])

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan2(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan2(i) for i in _INDEX] 

    def __init__(self, i1):
        self.name = f"GetSpan2({i1})"
        self.i1 = i1

    def check_next_btn(self, nxt_btn):
        if "GetSpan3" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        return get_span_mask_render(str1, pstate.past_buttons[-2:])

    def __call__(self, pstate):
        if "GetSpan1" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan3(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan3("Start"), GetSpan3("End")]

    def __init__(self, b1):
        self.name = f"GetSpan3({b1})"
        self.b1 = b1

    def check_next_btn(self, nxt_btn):
        if "GetSpan4" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        return get_span_mask_render(str1, pstate.past_buttons[-3:])

    def __call__(self, pstate):
        if "GetSpan2" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan4(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan4(name) for name in _POSSIBLE_R.keys()]

    def __init__(self, rname):
        self.name = f"GetSpan4({rname})"
        self.rname = rname
        self.r2 = _POSSIBLE_R[rname]

    def check_next_btn(self, nxt_btn):
        if "GetSpan5" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        return get_span_mask_render(str1, pstate.past_buttons[-4:])

    def __call__(self, pstate):
        if "GetSpan3" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan5(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan5(i) for i in _INDEX] 

    def __init__(self, i2):
        self.name = f"GetSpan5({i2})"
        self.i2 = i2

    def check_next_btn(self, nxt_btn):
        if "GetSpan6" not in nxt_btn.name:
            raise ButtonSeqError

    def str_masks_to_np(self, str1, pstate):
        return get_span_mask_render(str1, pstate.past_buttons[-5:])

    def __call__(self, pstate):
        if "GetSpan4" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan6(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan6("Start"), GetSpan6("End")]

    def __init__(self, b2):
        self.name = f"GetSpan6({b2})"
        self.b2 = b2

    def f(self, input_str, r1, i1, b1, r2, i2, b2):
        """
        sorry
        """
        return input_str[[m.end() for m in re.finditer(r1[0], input_str)][i1] if b1 == "End" else [m.start() for m in re.finditer(r1[0], input_str)][i1] : [m.end() for m in re.finditer(r2[0], input_str)][i2] if b2 == "End" else [m.start() for m in re.finditer(r2[0], input_str)][i2]]

    def __call__(self, pstate):
        if "GetSpan5" not in pstate.past_buttons[-1].name:
            raise ButtonSeqError
        r1 = pstate.past_buttons[-5].r1
        i1 = pstate.past_buttons[-4].i1
        b1 = pstate.past_buttons[-3].b1
        r2 = pstate.past_buttons[-2].r2
        i2 = pstate.past_buttons[-1].i2
        b2 = self.b2
        scratch_new = [self.f(x, r1, i1, b1, r2, i2, b2) for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class Const(Button):

    @staticmethod
    def generate_buttons():
        return [Const(c) for c in _CHARACTER]

    def __init__(self, c):
        self.name = f"Const({c})"
        self.c = c

    def __call__(self, pstate):
        scratch_new = [self.c for x in pstate.scratch]
        check_change(pstate.scratch, scratch_new)
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

ALL_BUTTS_TYPES = [ToCase,
                  Replace1,
                  Replace2,
                  SubStr1,
                  SubStr2,
                  GetToken1,
                  GetToken2,
                  GetUpTo,
                  GetFrom,
                  GetFirst1,
                  GetFirst2,
                  GetAll,
                  GetSpan1,
                  GetSpan2,
                  GetSpan3,
                  GetSpan4,
                  GetSpan5,
                  GetSpan6,
                  Const,
                  Commit,
                  ]

ALL_BUTTS = [x for butt_type in ALL_BUTTS_TYPES for x in butt_type.generate_buttons()]

ALL_BUTTS_NAME_MAP = {butt.name: ALL_BUTTS.index(butt) for butt in ALL_BUTTS}

class ROBENV:

    def __init__(self, inputs, outputs, 
                 render_kind={'render_scratch' : 'yes',
                              'render_past_buttons' : 'no'}):
        self.inputs, self.outputs = inputs, outputs
        self.verbose = False
        self.render_kind = render_kind

    def reset(self):
        self.done = False
        self.pstate = RobState.new(self.inputs, self.outputs)
        first_ob = self.pstate.to_np(self.render_kind)
        self.last_step = first_ob, 0.0, self.done
        return first_ob

    def copy(self):
        to_ret = ROBENV(self.inputs, self.outputs, self.render_kind)
        to_ret.done = self.done
        to_ret.pstate = self.pstate.copy()
        to_ret.last_step = self.last_step
        return to_ret

    def step(self, btn_action):
        try:
            self.pstate = btn_action(self.pstate)
            state_ob = self.pstate.to_np(self.render_kind)
            # check sequence other way around
            if len(self.pstate.past_buttons) >= 2:
                prev_btn, cur_btn = self.pstate.past_buttons[-2:]
                prev_btn.check_next_btn(cur_btn)
        except (IndexError, ButtonSeqError, CommitPrefixError, NoChangeError) as e:
        #except (IndexError, ButtonSeqError, NoChangeError) as e: #may also be able to get rid of noChangeError
            if self.verbose:
                print ("CATCHING")
                print ("error ", e)
                print(traceback.format_exc())
            self.done = True
            self.last_step = RobState.crash_state_np(self.render_kind), -1.0, True
            return self.last_step

        reward = 0.0 if self.pstate.committed != self.pstate.outputs else 1.0
        done = False if reward == 0.0 else True

        n_commits = len([x for x in self.pstate.past_buttons if x.name == "Commit"])

        if n_commits == N_EXPRS:
            done = True

        self.done = done
        self.last_step = state_ob, reward, done
        return self.last_step

class RepeatAgent:

    def __init__(self, btns):
        self.btns = btns
        self.idx = -1

    def act(self, state):
        self.idx += 1
        if self.idx >= len(self.btns):
            return Commit()
        return self.btns[self.idx]

def get_rollout(env, agent, max_iter):
    trace = []
    s = env.reset()
    for i in range(max_iter):
        
        if random.random() < 0.5:
            env = env.copy()
        a = agent.act(s)
        ss, r, done = env.step(a)
        trace.append((s, a, r, ss, str(env.pstate.scratch[0])))
        s = ss
        if done:
            break
    return trace

# ===================== UTILS ======================
class ButtonSeqError(Exception):
    """placeholder for button sequence error"""
    pass

class CommitPrefixError(Exception):
    """placeholder for commit mess up a prefix"""
    pass

class NoChangeError(Exception):
    """make sure we did not commit empty stuff"""
    pass

def check_change(old, new):
    if str(old) == str(new):
        raise NoChangeError

def get_span_mask_render(str1, span_btns):
    """
    span buttons starts from GetSpan1 until wherever
    """
    def render_span1(span1_btn):
        def render(past_mask):
            str_masks = Button.str_masks_to_np_default()
            # enumerate over all the regex masks
            p = list(re.finditer(span1_btn.r1[0], str1))
            for i, m in enumerate(p[:max(_INDEX)]):
                str_masks[i][m.start():] = 1
            return str_masks
        return render

    def render_span2(span2_btn):
        def render(past_mask):
            ret_mask = Button.str_masks_to_np_default()
            # the selected mask . . . 
            mask_sel = past_mask[span2_btn.i1]
            ret_mask[-1] = mask_sel
            return ret_mask
        return render

    def render_span3(span3_btn):
        def render(past_mask):
            span1_btn = span_btns[0]
            span2_btn = span_btns[1]
            # MAX UR IN CHARGE I GO GET KOFE
            r1 = span1_btn.r1
            i1 = span2_btn.i1

            m = list(re.finditer(r1[0], str1))[i1]

            if span3_btn.b1 == "End":
                past_mask[-1][m.start():m.end()] = 0
            return past_mask
        return render

    def render_span4(span4_btn):
        def render(past_mask):
            #str_masks = Button.str_masks_to_np_default()
            # enumerate over all the regex masks
            p = list(re.finditer(span4_btn.r2[0], str1))
            for i, m in enumerate(p[:max(_INDEX)]):
                past_mask[i][:m.end()] = 1
            return past_mask
        return render

    def render_span5(span5_btn):
        def render(past_mask):
            ret_mask = Button.str_masks_to_np_default()
            # the selected mask . . . 
            mask_sel = past_mask[span5_btn.i2]
            ret_mask[-1] = past_mask[-1]*mask_sel
            return ret_mask
        return render

    all_renders = [render_span1, render_span2, render_span3, render_span4, render_span5]
    all_renders = [factory(btn) for factory, btn in zip(all_renders, span_btns)]
    
    ret = None
    for render in all_renders:
        ret = render(ret)

    return ret

def apply_fs(pstate, funcs):
    if funcs == []:
        return pstate
    else:
        last_state = apply_fs(pstate, funcs[:-1])
        return funcs[-1](last_state)
