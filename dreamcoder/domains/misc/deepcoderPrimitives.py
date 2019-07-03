from dreamcoder.program import Primitive, prettyProgram
from dreamcoder.grammar import Grammar
from dreamcoder.type import tlist, tint, arrow, baseType #, t0, t1, t2

#from functools import reduce


#todo
int_to_int = baseType("int_to_int")
int_to_bool = baseType("int_to_bool")
int_to_int_to_int = baseType("int_to_int_to_int")


#deepcoderPrimitives
Null = 300 #or perhaps something else, like "an integer outside the working range"?

def _head(xs): return xs[0] if len(xs)>0 else Null
def _tail(xs): return xs[-1] if len(xs)>0 else Null
def _take(n): return lambda xs: xs[:n]
def _drop(n): return lambda xs: xs[n:]
def _access(n): return lambda xs: xs[n] if n>=0 and len(xs)>n else Null
def _minimum(xs): return min(xs) if len(xs)>0 else Null
def _maximum(xs): return max(xs) if len(xs)>0 else Null
def _reverse(xs): return list(reversed(xs))
def _sort(xs): return sorted(xs)
def _sum(xs): return sum(xs)

#higher order:
def _map(f): return lambda l: list(map(f, l)) 
def _filter(f): return lambda l: list(filter(f, l))
def _count(f): return lambda l: len([x for x in l if f(x)])
def _zipwith(f): return lambda xs: lambda ys: [f(x)(y) for (x, y) in zip(xs, ys)]
def _scanl1(f): 
    def _inner(xs):
        ys = []
        if len(xs) > 0:
            ys.append(xs[0])
            for i in range(1, len(xs)):
                ys.append( f(ys[i-1])(xs[i]))
        return ys
    return _inner

#int to int:
def _succ(x): return x+1
def _pred(x): return x-1
def _double(x): return x*2
def _half(x): return int(x/2)
def _negate(x): return -x
def _square(x): return x**2
def _triple(x): return x*3
def _third(x): return int(x/3)
def _quad(x): return x*4
def _quarter(x): return int(x/4)

#int to bool:
def _pos(x): return x>0
def _neg(x): return x<0
def _even(x): return x%2==0
def _odd(x): return x%2==1

#int to int to int:
def _add(x): return lambda y: x + y
def _sub(x): return lambda y: x - y
def _mult(x): return lambda y: x * y
def _min(x): return lambda y: _minimum([x,y])
def _max(x): return lambda y: _maximum([x,y])

def deepcoderPrimitives():
    return [
        Primitive("HEAD", arrow(tlist(tint), tint), _head), 
        Primitive("LAST", arrow(tlist(tint), tint), _tail),
        Primitive("TAKE", arrow(tint, tlist(tint), tlist(tint)), _take),
        Primitive("DROP", arrow(tint, tlist(tint), tlist(tint)), _drop),
        Primitive("ACCESS", arrow(tint, tlist(tint), tint), _access),
        Primitive("MINIMUM", arrow(tlist(tint), tint), _minimum),
        Primitive("MAXIMUM", arrow(tlist(tint), tint), _maximum),
        Primitive("REVERSE", arrow(tlist(tint), tlist(tint)), _reverse),
        Primitive("SORT", arrow(tlist(tint), tlist(tint)), _sort),
        Primitive("SUM", arrow(tlist(tint), tint), _sum)
        ] + [
        Primitive("MAP", arrow(int_to_int, tlist(tint), tlist(tint)), _map), #is this okay???
        Primitive("FILTER", arrow(int_to_bool, tlist(tint), tlist(tint)), _filter), #is this okay???
        Primitive("COUNT", arrow(int_to_bool, tlist(tint), tint), _count), #is this okay???
        Primitive("ZIPWITH", arrow(int_to_int_to_int, tlist(tint), tlist(tint), tlist(tint)), _zipwith), #is this okay???
        Primitive("SCANL1", arrow(int_to_int_to_int, tlist(tint), tlist(tint)), _scanl1), #is this okay???
        ] + [
        Primitive("INC", int_to_int, _succ),
        Primitive("DEC", int_to_int, _pred),
        Primitive("SHL", int_to_int, _double),
        Primitive("SHR", int_to_int, _half),
        Primitive("doNEG", int_to_int, _negate),
        Primitive("SQR", int_to_int, _square),
        Primitive("MUL3", int_to_int, _triple),
        Primitive("DIV3", int_to_int, _third),
        Primitive("MUL4", int_to_int, _quad),
        Primitive("DIV4", int_to_int, _quarter),
        ] + [
        Primitive("isPOS", int_to_bool, _pos),
        Primitive("isNEG", int_to_bool, _neg),
        Primitive("isEVEN", int_to_bool, _even),
        Primitive("isODD", int_to_bool, _odd),
        ] + [
        Primitive("+", int_to_int_to_int, _add),
        Primitive("-", int_to_int_to_int, _sub),
        Primitive("*", int_to_int_to_int, _mult),
        Primitive("MIN", int_to_int_to_int, _min),
        Primitive("MAX", int_to_int_to_int, _max)       
    ]

def OldDeepcoderPrimitives():
    return [
        Primitive("head", arrow(tlist(tint), tint), _head), 
        Primitive("tail", arrow(tlist(tint), tint), _tail),
        Primitive("take", arrow(tint, tlist(tint), tlist(tint)), _take),
        Primitive("drop", arrow(tint, tlist(tint), tlist(tint)), _drop),
        Primitive("access", arrow(tint, tlist(tint), tint), _access),
        Primitive("minimum", arrow(tlist(tint), tint), _minimum),
        Primitive("maximum", arrow(tlist(tint), tint), _maximum),
        Primitive("reverse", arrow(tlist(tint), tlist(tint)), _reverse),
        Primitive("sort", arrow(tlist(tint), tlist(tint)), _sort),
        Primitive("sum", arrow(tlist(tint), tint), _sum)
        ] + [
        Primitive("map", arrow(int_to_int, tlist(tint), tlist(tint)), _map), #is this okay???
        Primitive("filter", arrow(int_to_bool, tlist(tint), tlist(tint)), _filter), #is this okay???
        Primitive("count", arrow(int_to_bool, tlist(tint), tint), _count), #is this okay???
        Primitive("zipwith", arrow(int_to_int_to_int, tlist(tint), tlist(tint), tlist(tint)), _zipwith), #is this okay???
        Primitive("scanl1", arrow(int_to_int_to_int, tlist(tint), tlist(tint)), _scanl1), #is this okay???
        # ] + [
        # Primitive("succ", arrow(tint, tint), _succ),
        # Primitive("pred", arrow(tint, tint), _pred),
        # Primitive("double", arrow(tint, tint), _double),
        # Primitive("half", arrow(tint, tint), _half),
        # Primitive("neg", arrow(tint, tint), _neg),
        # Primitive("square", arrow(tint, tint), _square),
        # Primitive("triple", arrow(tint, tint), _triple),
        # Primitive("third", arrow(tint, tint), _third),
        # Primitive("quad", arrow(tint, tint), _quad),
        # Primitive("quarter", arrow(tint, tint), _quarter),
        # ] + [
        # Primitive("pos", arrow(tint, tbool), _pos),
        # Primitive("neg", arrow(tint, tbool), _neg),
        # Primitive("even", arrow(tint, tbool), _even),
        # Primitive("odd", arrow(tint, tbool), _odd),
        # ] + [
        # Primitive("add", arrow(tint, tint, tint), _add),
        # Primitive("sub", arrow(tint, tint, tint), _sub),
        # Primitive("mult", arrow(tint, tint, tint), _mult),
        # Primitive("min", arrow(tint, tint, tint), _min),
        # Primitive("max", arrow(tint, tint, tint), _max)
        ] + [
        Primitive("succ_fn", int_to_int, _succ),
        Primitive("pred_fn", int_to_int, _pred),
        Primitive("double_fn", int_to_int, _double),
        Primitive("half_fn", int_to_int, _half),
        Primitive("negate_fn", int_to_int, _negate),
        Primitive("square_fn", int_to_int, _square),
        Primitive("triple_fn", int_to_int, _triple),
        Primitive("third_fn", int_to_int, _third),
        Primitive("quad_fn", int_to_int, _quad),
        Primitive("quarter_fn", int_to_int, _quarter),
        ] + [
        Primitive("pos_fn", int_to_bool, _pos),
        Primitive("neg_fn", int_to_bool, _neg),
        Primitive("even_fn", int_to_bool, _even),
        Primitive("odd_fn", int_to_bool, _odd),
        ] + [
        Primitive("add_fn", int_to_int_to_int, _add),
        Primitive("sub_fn", int_to_int_to_int, _sub),
        Primitive("mult_fn", int_to_int_to_int, _mult),
        Primitive("min_fn", int_to_int_to_int, _min),
        Primitive("max_fn", int_to_int_to_int, _max)       
    ]

def deepcoderProductions():
    return [(0.0, prim) for prim in deepcoderPrimitives()]

def flatten_program(p):
    string = p.show(False)
    num_inputs = string.count('lambda')
    string = string.replace('lambda', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    #remove '_fn' (optional)
    for i in range(num_inputs):
        string = string.replace('$' + str(num_inputs-i-1),'input_' + str(i))
    string = string.split(' ')
    string = list(filter(lambda x: x is not '', string))
    return string

if __name__ == "__main__":
    #g = Grammar.uniform(deepcoderPrimitives())
    g = Grammar.fromProductions(deepcoderProductions(), logVariable=.9)
    request = arrow(tlist(tint), tint, tint)
    p = g.sample(request)
    print("request:", request)
    print("program:")
    print(prettyProgram(p))
    print("flattened_program:")
    flat = flatten_program(p)
    print(flat)

    #robustfill output = names from productions + input_0-2 or 3





    # # with open("/home/ellisk/om/ec/experimentOutputs/list_aic=1.0_arity=3_ET=1800_expandFrontier=2.0_it=4_likelihoodModel=all-or-nothing_MF=5_baseline=False_pc=10.0_L=1.0_K=5_rec=False.pickle", "rb") as handle:
    # #     b = pickle.load(handle).grammars[-1]
    # # print b

    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (empty? $0) empty (cons (+ (car $1) (car $0)) ($2 (cdr $1) (cdr $0)))))))")
    # t = arrow(tlist(tint), tlist(tint), tlist(tint))  # ,tlist(tbool))
    # print(g.logLikelihood(arrow(t, t), p))
    # assert False
    # print(b.logLikelihood(arrow(t, t), p))

    # # p = Program.parse("""(lambda (lambda
    # # (unfold 0
    # # (lambda (+ (index $0 $2) (index $0 $1)))
    # # (lambda (1+ $0))
    # # (lambda (eq? $0 (length $1))))))
    # # """)
    # p = Program.parse("""(lambda (lambda
    # (map (lambda (+ (index $0 $2) (index $0 $1))) (range (length $0))  )))""")
    # # .replace("unfold", "#(lambda (lambda (lambda (lambda (fix1 $0 (lambda (lambda (#(lambda (lambda (lambda (if $0 empty (cons $1 $2))))) ($1 ($3 $0)) ($4 $0) ($5 $0)))))))))").\
    # # replace("length", "#(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ ($1 (cdr $0)) 1))))))").\
    # # replace("forloop", "(#(lambda (lambda (lambda (lambda (fix1 $0 (lambda (lambda (#(lambda (lambda (lambda (if $0 empty (cons $1 $2))))) ($1 ($3 $0)) ($4 $0) ($5 $0))))))))) (lambda (#(eq? 0) $0)) $0 (lambda (#(lambda (- $0 1)) $0)))").\
    # # replace("inc","#(lambda (+ $0 1))").\
    # # replace("drop","#(lambda (lambda (fix2 $0 $1 (lambda (lambda (lambda (if
    # # (#(eq? 0) $1) $0 (cdr ($2 (- $1 1) $0)))))))))"))
    # print(p)
    # print(g.logLikelihood(t, p))
    # assert False

    # print("??")
    # p = Program.parse(
    #     "#(lambda (#(lambda (lambda (lambda (fix1 $0 (lambda (lambda (if (empty? $0) $3 ($4 (car $0) ($1 (cdr $0)))))))))) (lambda $1) 1))")
    # for j in range(10):
    #     l = list(range(j))
    #     print(l, p.evaluate([])(lambda x: x * 2)(l))
    #     print()
    # print()

    # print("multiply")
    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (eq? $0 0) 0 (+ $1 ($2 $1 (- $0 1)))))))")
    # print(g.logLikelihood(arrow(arrow(tint, tint, tint), tint, tint, tint), p))
    # print()

    # print("take until 0")
    # p = Program.parse("(lambda (lambda (if (eq? $1 0) empty (cons $1 $0))))")
    # print(g.logLikelihood(arrow(tint, tlist(tint), tlist(tint)), p))
    # print()

    # print("countdown primitive")
    # p = Program.parse(
    #     "(lambda (lambda (if (eq? $0 0) empty (cons (+ $0 1) ($1 (- $0 1))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint, tlist(tint)), arrow(
    #                 tint, tlist(tint))), p))
    # print(_fix(9)(p.evaluate([])))
    # print("countdown w/ better primitives")
    # p = Program.parse(
    #     "(lambda (lambda (if (eq0 $0) empty (cons (+1 $0) ($1 (-1 $0))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint, tlist(tint)), arrow(
    #                 tint, tlist(tint))), p))

    # print()

    # print("prepend zeros")
    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (eq? $1 0) $0 (cons 0 ($2 (- $1 1) $0))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint,
    #                 tlist(tint),
    #                 tlist(tint)),
    #             tint,
    #             tlist(tint),
    #             tlist(tint)),
    #         p))
    # print()
    # assert False

    # p = Program.parse(
    #     "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ 1 ($1 (cdr $0))))))))")
    # print(p.evaluate([])(list(range(17))))
    # print(g.logLikelihood(arrow(tlist(tbool), tint), p))

    # p = Program.parse(
    #     "(lambda (lambda (if (empty? $0) 0 (+ 1 ($1 (cdr $0))))))")
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tlist(tbool), tint), arrow(
    #                 tlist(tbool), tint)), p))

    # p = Program.parse(
    #     "(lambda (fix1 $0 (lambda (lambda (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))))")

    # print(p.evaluate([])(list(range(4))))
    # print(g.logLikelihood(arrow(tlist(tint), tint), p))

    # p = Program.parse(
    #     "(lambda (lambda (if (empty? $0) 0 (+ (car $0) ($1 (cdr $0))))))")
    # print(p)
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tlist(tint),
    #                 tint),
    #             tlist(tint),
    #             tint),
    #         p))

    # print("take")
    # p = Program.parse(
    #     "(lambda (lambda (lambda (if (eq? $1 0) empty (cons (car $0) ($2 (- $1 1) (cdr $0)))))))")
    # print(p)
    # print(
    #     g.logLikelihood(
    #         arrow(
    #             arrow(
    #                 tint,
    #                 tlist(tint),
    #                 tlist(tint)),
    #             tint,
    #             tlist(tint),
    #             tlist(tint)),
    #         p))
    # assert False

    # print(p.evaluate([])(list(range(4))))
    # print(g.logLikelihood(arrow(tlist(tint), tlist(tint)), p))

    # p = Program.parse(
    #     """(lambda (fix (lambda (lambda (match $0 0 (lambda (lambda (+ $1 ($3 $0))))))) $0))""")
    # print(p.evaluate([])(list(range(4))))
    # print(g.logLikelihood(arrow(tlist(tint), tint), p))
