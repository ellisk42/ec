"""Primitives designed for the RE2 domain of the Learning with Latent Language paper."""

from dreamcoder.program import Primitive, Program
from dreamcoder.grammar import Grammar
from dreamcoder.type import tint, tlist, arrow, baseType, tbool, t0
from dreamcoder.domains.list.listPrimitives import bootstrapTarget,  re2_listPrimitives_v1

import re


tfullstr = baseType("fullstr")
tsubstr = baseType("substr")

# Regex constants -- handled by constructing regex substrings
_rvowel = "(a|e|i|o|u)" 
_rconsonant = "[^aeiou]"
alpha_chars = [chr(ord('a') + j) for j in range(26)] 
def _rnot(s): return f"[^{s}]"
def _ror(s1): return lambda s2: f"(({s1})|({s2}))"
def _rconcat(s1): return lambda s2: s1 + s2  
                                
# Evaluate s1 as a regex against r2
def __ismatch(s1, s2):
    try:
        return re.fullmatch(re.compile(s1), s2) is not None 
    except e:
        return False
def _rmatch(s1) : return lambda s2: __ismatch(s1, s2)

# Splits s2 on regex s1 as delimiter, including the matches
def __regex_split(s1, s2):
    try:
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
def _rsplit(s1) : return lambda s2: __regex_split(s1, s2)

# Flattens list of substrings back into a string
def _rflatten(l): return "".join(l)
    
def _rtail(l) : return l[-1]
def _rappend(x) : return lambda l: l + [x]
def _rrevcdr(l) : return l[:-1]

# Strongly typed version.
def re2_vowel_consonant():
    return [Primitive("_rvowel", tsubstr, _rvowel) +
            Primitive("_rconsonant", tsubstr, _rconsonant)]
            
def re2_primitives_v1():
    # [a-z] + [.]
    regex_constants = [Primitive("_rdot", tsubstr, "."),
                       Primitive("_emptystr", tsubstr, "")] + \
                      [Primitive("_%s" % c, tsubstr, c) for c in alpha_chars]
                              
    return regex_constants + \
         [
            Primitive("_rnot", arrow(tsubstr, tsubstr), _rnot),
            Primitive("_ror", arrow(tsubstr, tsubstr, tsubstr), _ror),
            Primitive("_rconcat", arrow(tsubstr, tsubstr, tsubstr), _rconcat),
            
            Primitive("_rmatch", arrow(tsubstr, tsubstr, tbool), _rmatch),
            Primitive("_rsplit", arrow(tfullstr, tsubstr, tlist(tsubstr)), _rsplit),
            Primitive("_rflatten", arrow(tlist(tsubstr), tfullstr), _rflatten),
            Primitive("_rtail", arrow(tlist(t0), t0), _rtail),
            Primitive("_rappend", arrow(tlist(t0), t0, tlist(t0)), _rappend),
            Primitive("_rrevcdr", arrow(tlist(t0), tlist(t0)), _rrevcdr),
         ] +  re2_listPrimitives_v1()

def re2_primitives_main():
    re2_primitives_v1()
    bootstrapTarget()
    def check_true(name, raw, input):
        p = Program.parse(raw)
        pass_test = "[T]" if p.evaluate([])(input) else "[F]"
        print(f"{pass_test} {name} : {input} : {p.evaluate([])(input)}")
    
    def check_equal(name, raw, input, gold):
        p = Program.parse(raw)
        output = p.evaluate([])(input)
        print(f"{name} in: {input} | out: {output} | gold: {gold}")
        
    # Simple matches on one string.
    
    SIMPLE_MATCHES = True
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
        
    # Single string manipulations
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

