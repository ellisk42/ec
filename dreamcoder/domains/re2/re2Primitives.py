"""Primitives designed for the RE2 domain of the Learning with Latent Language paper."""

from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import tint, tlist, arrow, baseType, tbool, t0, t1
from dreamcoder.domains.text.textPrimitives import re2_text_primitives, re2_text_4_letter, re2_text_6_letter, re2_text_characters
from dreamcoder.domains.list.listPrimitives import re2_list_v0, _cons, _car, _cdr, _map, _if
import re


tfullstr = baseType("tfullstr")
tsubstr = baseType("tsubstr")

### Regex constants -- handled by constructing regex substrings
def re2_constants(prim_name):
    rdot = Primitive("_rdot", tsubstr, ".")
    rempty = Primitive("_rempty", tsubstr, "")
    alpha_chars = [chr(ord('a') + j) for j in range(26)] 
    
    chars = prim_name.split("re2_chars_")[-1]
    if chars == "None": chars = None
    if chars is None:
        chars = alpha_chars
    else:
        chars = [chr(ord(c)) for c in list(chars)]
    char_constants = [Primitive("_%s" % c, tsubstr, c) for c in chars]
    
    return [rdot, rempty] + char_constants
def re2_vowel_consonant_primitives():
    _rvowel = Primitive("_rvowel", tsubstr, "(a|e|i|o|u)") 
    _rconsonant = Primitive("_rconsonant", tsubstr,  "[^aeiou]")
    return [_rvowel, _rconsonant]

### Basic regex substring manipulations
def _rnot(s): return f"[^{s}]"
def _ror(s1): return lambda s2: f"(({s1})|({s2}))"
def _rconcat(s1): return lambda s2: s1 + s2  
re2_rnot = Primitive("_rnot", arrow(tsubstr, tsubstr), _rnot)
re2_ror = Primitive("_ror", arrow(tsubstr, tsubstr, tsubstr), _ror)
re2_rconcat = Primitive("_rconcat", arrow(tsubstr, tsubstr, tsubstr), _rconcat)

### Regex matching.                            
# Evaluate s1 as a regex against r2
def __ismatch(s1, s2):
    try:
        return re.fullmatch(re.compile(s1), s2) is not None 
    except e:
        return False
def __regex_split(s1, s2):
    # Splits s2 on regex s1 as delimiter, including the matches
    try:
        # Special case -- we override splitting on "" to be splitting on "."
        # to match OCaml.
        if len(s1) == 0: s1 = "."
        ret = []
        remaining = s2
        m = re.search(re.compile(s1), remaining)
        while m is not None:
            prefix = remaining[0:m.start()]
            if len(prefix) > 0:
                ret.append(prefix)
            ret.append(remaining[m.start():m.end()])
            remaining = remaining[m.end():]
            m = re.search(re.compile(s1), remaining)
        if len(remaining) > 0:
            ret.append(remaining)
        return ret        
    except e:
        return [s2]
def _rmatch(s1) : return lambda s2: __ismatch(s1, s2)
def _rsplit(s1) : return lambda s2: __regex_split(s1, s2)
def _rflatten(l): return "".join(l) # Flattens list of substrings back into a string
re2_rmatch = Primitive("_rmatch", arrow(tsubstr, tsubstr, tbool), _rmatch)
re2_rsplit = Primitive("_rsplit", arrow(tsubstr, tfullstr, tlist(tsubstr)), _rsplit)
re2_rflatten = Primitive("_rflatten", arrow(tlist(tsubstr), tfullstr), _rflatten)

### List operators
def _rtail(l) : return l[-1]
def _rappend(x) : return lambda l: l + [x]
def _rrevcdr(l) : return l[:-1]

re2_if = Primitive("if", arrow(tbool, t0, t0, t0), _if)
re2_cons = Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons)
re2_car = Primitive("car", arrow(tlist(t0), t0), _car)
re2_cdr = Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr)
re2_map = Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map)
re2_rtail = Primitive("_rtail", arrow(tlist(tsubstr), tsubstr), _rtail)
re2_rappend = Primitive("_rappend", arrow(t0, tlist(t0), tlist(t0)), _rappend)
re2_rrevcdr = Primitive("_rrevcdr", arrow(tlist(t0), tlist(t0)), _rrevcdr)
        
def re2_bootstrap_v1_primitives():
    return [re2_rnot, re2_ror, re2_rconcat] \
        +  [re2_rmatch, re2_rsplit, re2_rflatten] \
        +  [re2_rtail, re2_rappend, re2_rrevcdr] \
        +  [re2_if, re2_cons, re2_car, re2_cdr, re2_map]

def re2_test_primitives():
    return [re2_rsplit, re2_rflatten] \
        +  [re2_ror,]

def load_re2_primitives(primitive_names):
    prims = []
    type_request = None
    for pname in primitive_names:
        if pname.startswith("re2_chars"):
            prims += re2_constants(pname)
        elif pname == 're2_test':
            prims += re2_test_primitives()
            type_request = "tfullstr"
        elif pname == 're2_bootstrap_v1_primitives':
            prims += re2_bootstrap_v1_primitives()
            type_request = "tfullstr"
        elif pname == 're2_vowel_consonant_primitives':
            prims += re2_vowel_consonant_primitives()
            type_request = "tfullstr"
        
        # Old primitive sets
        elif pname == 're2_primitives': 
            prims += re2_text_primitives
            prims += re2_list_v0()
            type_request = "list_tcharacter"
        elif pname == 're2_4_letter':
            prims += re2_text_4_letter
            prims += re2_list_v0()
            type_request = "list_tcharacter"
        elif pname == 're2_6_letter':
            prims += re2_text_6_letter
            prims += re2_list_v0()
            type_request = "list_tcharacter"
    return prims, type_request
    
def re2_primitives_main():
    def check_true(name, raw, input):
        p = Program.parse(raw)
        pass_test = "[T]" if p.evaluate([])(input) else "[F]"
        print(f"{pass_test} {name} : {input} : {p.evaluate([])(input)}")
    
    def check_equal(name, raw, input, gold):
        p = Program.parse(raw)
        output = p.evaluate([])(input)
        print(f"{name} in: {input} | out: {output} | gold: {gold}")
    
    DEBUG_EMPTY = False
    if DEBUG_EMPTY:
        # Debugging the 'split empty' phenomenon
        raw = "(lambda (_rflatten (map (lambda _k) (_rsplit _rempty $0))))"
        input_str = "aaa"
        check_true("basic input", raw, input_str)

        raw = "(lambda $0)"
        raw = "(lambda (car (cdr (map (lambda $1) (_rrevcdr (_rsplit _r $0))))))"
        
        # Simple matches on one string.
        raw = "(lambda (_rflatten (_rsplit _rdot $0)))"
        input_str = "aaa"
        check_true("basic input", raw, input_str)
    
    
    SIMPLE_MATCHES = False
    if SIMPLE_MATCHES:
        input_str = "t"
        raw = "(lambda $0)"
        check_true("basic input", raw, input_str)
        
        input_str = "t"
        raw = "(lambda (_rmatch _rdot $0))"
        check_true("match .", raw, input_str)
        input_str = "tt"
        raw = "(lambda (_rmatch _rdot $0))"
        check_true("match .", raw, input_str)
        
        input_str = "ab"
        raw = "(lambda (_rmatch (_rconcat _a _b) $0))"
        check_true("match ab", raw, input_str)
        input_str = "ab"
        raw = "(lambda (_rmatch (_rconcat _rdot _b) $0))"
        check_true("match .b", raw, input_str)
        
        input_str = "abc"
        raw = "(lambda (_rmatch (_rconcat (_rconcat _rdot _b) _c) $0))"
        check_true("match .bc", raw, input_str)
        
        input_str = "b"
        raw = "(lambda (_rmatch (_ror _b _c) $0))"
        check_true("match (b|c)", raw, input_str)
        input_str = "bd"
        raw = "(lambda (_rmatch (_rconcat (_ror _b _c) _d) $0))"
        check_true("match ((b|c))d", raw, input_str)
        
        input_str = "b"
        raw = "(lambda (_rmatch (_rnot (_rconcat _a _e)) $0))"
        check_true("match [^ae]", raw, input_str)

        raw = "(lambda (_rmatch (_rconcat (_rnot (_rconcat _a _e)) _d) $0))"
        check_true("match [^ae]d", raw, "bd")
        check_true("match [^ae]d", raw, "ad")
        check_true("match [^ae]d", raw, "be")
        
    # Splitting on a Boolean.
    DEBUG_BOOLEAN = True
    if DEBUG_BOOLEAN:
        raw = "(lambda  (if (_rmatch (_ror _a _e) $0) _f $0)   )"
        check_equal("replace a -> f", raw, "a", "f")
        
        replace_ae = "(lambda  (if (_rmatch (_ror _a _e) $0) _f $0)   )"
        input = "hella"
        raw = f"(lambda (_rflatten (map  {replace_ae}  (_rsplit (_ror _a _e) $0) ) ))"
        # # Replace vowels 
        check_equal("replace a|e -> f", raw, input, "hfllf")
        # Replace any letter vowel 
        
        # CHeck replace vowels
        input = "aeioutestaeioutest"
        replace_vowel = "(lambda  (if (_rmatch _rvowel $0) _f $0)   )"
        raw = f"(lambda (_rflatten (map  {replace_vowel}  (_rsplit _a $0) ) ))"
        check_equal("replace a|e -> f", raw, input, "hfllf")
    # Replace a vowel letter sequence

    
    # Single string manipulations
    if False:
        raw = "(lambda  (if (_rmatch (_rconcat _t _e) $0) _f $0)   )"
        check_equal("replace te -> f", raw, "te", "f")
        check_equal("replace te -> f", raw, "zee", "zee")
        
        # Replace match
        replace_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) _f $0)   )"
        raw = f"(lambda (_rflatten (map  {replace_te}  (_rsplit (_rconcat _t _e) $0) ) ))"
        check_equal("replace te -> f", raw, "tehellote", "fhellof")
        # Prepend X to match
        prepend_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) (_rconcat _f $0) $0)   )"
        raw = f"(lambda (_rflatten (map  {prepend_te}  (_rsplit (_rconcat _t _e) $0) ) ))"
        check_equal("prepend te -> f", raw, "tehellote", "ftehellofte")
        # Postpend X to match
        postpend_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) (_rconcat $0 _f) $0)   )"
        raw = f"(lambda (_rflatten (map  {postpend_te}  (_rsplit (_rconcat _t _e) $0) ) ))"
        check_equal("postpend te -> f", raw, "tehellote", "tefhellotef")
        
        # Remove match
        remove_te = "(lambda  (if (_rmatch (_rconcat _t _e) $0) _emptystr $0)   )"
        raw = f"(lambda (_rflatten (map  {remove_te }  (_rsplit (_rconcat _t _e) $0) ) ))"
        check_equal("prepend te -> f", raw, "teheltelote", "hello")
        
        # Match at start
        raw = f"(lambda ((lambda (_rflatten (cons ({replace_te} (car $0)) (cdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))"
        check_equal("replace te -> f", raw, "teheltelote", "fheltelote")
        
        raw = f"(lambda ((lambda (_rflatten (cons ({prepend_te} (car $0)) (cdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))"
        check_equal("prepend te -> f", raw, "tehellote", "ftehellote")
        raw = f"(lambda ((lambda (_rflatten (cons ({postpend_te} (car $0)) (cdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))"
        check_equal("postpend te -> f", raw, "tehellote", "tefhellote")
        
        # Match at end
        raw = f"(lambda ((lambda (_rflatten (_rappend ({replace_te} (_rtail $0)) (_rrevcdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))"
        check_equal("replace te -> f", raw, "teheltelote", "teheltelof")
        
        raw = f"(lambda ((lambda (_rflatten (_rappend ({prepend_te} (_rtail $0)) (_rrevcdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))"
        check_equal("prepend te -> f", raw, "teheltelote", "teheltelofte")
        
        raw = f"(lambda ((lambda (_rflatten (_rappend ({postpend_te} (_rtail $0)) (_rrevcdr $0)) )) (_rsplit (_rconcat _t _e) $0) ))"
        check_equal("postpend te -> f", raw, "teheltelote", "teheltelotef")

