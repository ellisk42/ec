from task import *
from type import *

import math
import sys
import random
from random import choice
from collections import OrderedDict, Counter
from itertools import takewhile,dropwhile
import numpy as np

PRIMES = [2,3,5,7,11,13,17,19,23,29] # all the primes I want to write down right now
SQUARES = [x*x for x in range(10)]

def map_predicate(predicate,l):
    return [ z
             for x in l
             for z in [x, int(predicate(x))] ]

# return N random lists
def random_lists(n, minlength=None):
    ls = []
    if minlength == None:
        while len(ls) < n:
            l = [choice(range(10)) for _ in range(choice(range(15)))]
            if not str(l) in map(str, ls):
                ls.append(l)
    else:
        while len(ls) < n:
            l = [choice(range(10)) for _ in range(minlength + choice(range(15 - minlength)))]
            if not str(l) in map(str, ls):
                ls.append(l)
    return ls

global_problems = []
def problem(description, examples):
    x,y = examples[0]
    if isinstance(x,list): inputType = tlist(tint)
    elif isinstance(x,int): inputType = tint
    else: assert False
    if isinstance(y,list): outputType = tlist(tint)
    elif isinstance(y,int): outputType = tint
    else: assert False

    t = RegressionTask(description, arrow(inputType,outputType),
                       [((x,),y) for x,y in examples ],
                       features = examples,
                       cache = False)
    global_problems.append(t)
    
# Element-wise add
def f_elementwise_add(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Add %d to each element' % e,
                [(x, [j + e for j in x]) for x in random_lists(num_ex)])

# Element-wise product
def f_elementwise_prod(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Multiply %d to each element' % e,
                [(x, [int(j * e) for j in x]) for x in random_lists(num_ex)])

# Append an element to a list
def f_append(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Append element %d to array' % e,
                [(x, x + [e]) for x in random_lists(num_ex)])

# Prepend an element to a list
def f_prepend(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Prepend element %d to array' % e,
                [(x, [e] + x) for x in random_lists(num_ex)])

# Remove duplicate elements from a list
def f_remove_duplicates(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Remove duplicates from array',
                [(x, list(OrderedDict.fromkeys(x))) for x in random_lists(num_ex)])

# Duplicate each element of a list
def f_duplicate_each_element(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Duplicate each element in an array',
                [(x, [x[j//2] for j in range(len(x)*2)]) for x in random_lists(num_ex)])

# Remove the even numbers from a list
def f_remove_even(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Remove even numbers from an array',
                [(x, list(filter(lambda x : x % 2 == 1, x))) for x in random_lists(num_ex)])

# Remove the odd numbers from a list
def f_remove_odd(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Remove odd numbers from an array',
                [(x, list(filter(lambda x : x % 2 == 0, x))) for x in random_lists(num_ex)])

# Remove the % N numbers from a list
def f_remove_modulo_N(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Remove numbers divisible by %d from an array' % e,
                [(x, list(filter(lambda x : x % e != 0, x))) for x in random_lists(num_ex)])

# Return the even numbers from a list
def f_return_even(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return even numbers from an array',
                [(x, list(filter(lambda x : x % 2 == 0, x))) for x in random_lists(num_ex)])

# Return the odd numbers from a list
def f_return_odd(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return odd numbers from an array',
                [(x, list(filter(lambda x : x % 2 == 1, x))) for x in random_lists(num_ex)])

# Return the % N numbers from a list
def f_return_modulo_N(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Return numbers divisible by %d from an array' % e,
                [(x, list(filter(lambda x : x % e == 0, x))) for x in random_lists(num_ex)])

# Remove the Nth-largest element in a list
def f_remove_Nth_largest(num_progs, num_ex=10):
    tot = 0
    while tot < num_progs:
        try:
            e = random.randint(1, 9)
            if e % 10 == 1:
                problem('Remove %d' % e + 'st largest number from an array',
                        [(x, filter(lambda j : j != np.partition(x, -1*e)[-1*e], x)) for x in random_lists(num_ex, e)])
            elif e % 10 == 2:
                problem('Remove %d' % e + 'nd largest number from an array',
                        [(x, filter(lambda j : j != np.partition(x, -1*e)[-1*e], x)) for x in random_lists(num_ex, e)])
            elif e % 10 == 3:
                problem('Remove %d' % e + 'rd largest number from an array',
                        [(x, filter(lambda j : j != np.partition(x, -1*e)[-1*e], x)) for x in random_lists(num_ex, e)])
            else:
                problem('Remove %d' % e + 'th largest number from an array',
                        [(x, filter(lambda j : j != np.partition(x, -1*e)[-1*e], x)) for x in random_lists(num_ex, e)])
            tot += 1
        except:
            pass

# Remove the Nth element in a list
def f_remove_Nth_element(num_progs, num_ex=10):
    tot = 0
    while tot < num_progs:
        try:
            e = random.randint(1, 9)
            if e % 10 == 1:
                problem('Remove %d' % e + 'st element from an array',
                        [(x, x[:e-1] + x[e:]) for x in random_lists(num_ex, e)])
            if e % 10 == 2:
                problem('Remove %d' % e + 'nd element from an array',
                        [(x, x[:e-1] + x[e:]) for x in random_lists(num_ex, e)])
            if e % 10 == 3:
                problem('Remove %d' % e + 'rd element from an array',
                        [(x, x[:e-1] + x[e:]) for x in random_lists(num_ex, e)])
            else:
                problem('Remove %d' % e + 'th element from an array',
                        [(x, x[:e-1] + x[e:]) for x in random_lists(num_ex, e)])
            tot += 1
        except:
            pass

# Return the Nth-largest element in a list
def f_return_Nth_largest(num_progs, num_ex=10):
    tot = 0
    while tot < num_progs:
        try:
            e = random.randint(1, 9)
            if e % 10 == 1:
                problem('Return %d' % e + 'st largest number from an array',
                        [(x, [np.sort(x)[-1*e]]) for x in random_lists(num_ex, e)])
            elif e % 10 == 2:
                problem('Return %d' % e + 'nd largest number from an array',
                        [(x, [np.sort(x)[-1*e]]) for x in random_lists(num_ex, e)])
            elif e % 10 == 3:
                problem('Return %d' % e + 'rd largest number from an array',
                        [(x, [np.sort(x)[-1*e]]) for x in random_lists(num_ex, e)])
            else:
                problem('Return %d' % e + 'th largest number from an array',
                        [(x, [np.sort(x)[-1*e]]) for x in random_lists(num_ex, e)])

            tot += 1
        except:
            pass

# Return the Nth element in a list
def f_return_Nth_element(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        if e % 10 == 1:
            problem('Return %d' % e + 'st number from an array',
                    [(x, x[e-1]) for x in random_lists(num_ex, e)])
        elif e % 10 == 2:
            problem('Return %d' % e + 'nd number from an array',
                    [(x, x[e-1]) for x in random_lists(num_ex, e)])
        elif e % 10 == 3:
            problem('Return %d' % e + 'rd number from an array',
                    [(x, x[e-1]) for x in random_lists(num_ex, e)])
        else:
            problem('Return %d' % e + 'th number from an array',
                    [(x, x[e-1]) for x in random_lists(num_ex, e)])

# Check whether an item is a member of a list
def f_check_element(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Check if %d is a member of the array' % e,
                [(x, [0] if e in x else [1]) for x in random_lists(num_ex)])

# Replace every item in a list with the Nth item
def f_replace_all_with_Nth(num_progs, num_ex=10):
    tot = 0
    while tot < num_progs:
        try:
            e = random.randint(1, 9)
            if e % 10 == 1:
                problem('Replace every element in an array with the %d' % e + 'st element',
                        [(x, [x[e-1] for _ in range(len(x))]) for x in random_lists(num_ex, e)])
            if e % 10 == 2:
                problem('Replace every element in an array with the %d' % e + 'nd element',
                        [(x, [x[e-1] for _ in range(len(x))]) for x in random_lists(num_ex, e)])
            if e % 10 == 3:
                problem('Replace every element in an array with the %d' % e + 'rd element',
                        [(x, [x[e-1] for _ in range(len(x))]) for x in random_lists(num_ex, e)])
            else:
                problem('Replace every element in an array with the %d' % e + 'th element',
                        [(x, [x[e-1] for _ in range(len(x))]) for x in random_lists(num_ex, e)])

            tot += 1
        except:
            pass

# Intersperse X into array
def f_intersperse_X(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Intersperse %d into array between each element' % e,
                [(x, [e if j % 2 == 1 else x[j//2] for j in range(len(x)*2)]) for x in random_lists(num_ex)])

# Reverse a list
def f_reverse_array(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Reverse the array',
                [(x, x[::-1]) for x in random_lists(num_ex)])

# Sort a list
def f_sort_array(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Sort the array',
                [(x, sorted(x)) for x in random_lists(num_ex)])

# Reverse-sort a list
def f_reverse_sort_array(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Sort the array in reverse order',
                [(x, sorted(x, reverse=True)) for x in random_lists(num_ex)])

# Swap first half and second half
def f_swap_first_and_second_half(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Swap the first and second halves of the array',
                [(x, x[len(x)/2:] + x[:len(x)/2]) for x in random_lists(num_ex)])

# Shift all elements in a list by X (negative or positive)
def f_shift_elements(num_progs, num_ex=10):
    for i in range(num_progs):
        e = random.randint(1, 9)
        problem('Shift all elements to the right in an array by %d' % e,
                [(x, [x[(e + j) % len(x)] for j in range(len(x))]) for x in random_lists(num_ex)])

# Return sum
def f_sum_array(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return the sum of the elements in the array',
                [(x, [sum(x)]) for x in random_lists(num_ex)])

# Return product
def f_prod_array(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return the product of the elements in the array',
                [(x, [np.prod(x)]) for x in random_lists(num_ex)])

# Return the length of a list
def f_length_array(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return the length of the array',
                [(x, len(x)) for x in random_lists(num_ex)])

def f_count_head_in_tail(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Count head in tail',
                [(x, [Counter(x)[x[0]] - 1]) for x in random_lists(num_ex, 1)])

def f_check_if_even_1(num_progs, num_ex=10):
    predicate = lambda x : x % 2 == 0
    for i in range(num_progs):
        problem("Check if elements are even",
                [(l, map_predicate(predicate, l)) for l in random_lists(num_ex)])

def f_check_if_odd_1(num_progs, num_ex=10):
    predicate = lambda x : x % 2 == 1
    for i in range(num_progs):
        problem("Check if elements are odd",
                [(l, map_predicate(predicate, l)) for l in random_lists(num_ex)])

def f_check_if_prime_1(num_progs, num_ex=10):
    predicate = lambda x : x in PRIMES
    for i in range(num_progs):
        problem("Check if elements are prime",
                [(l, map_predicate(predicate, l)) for l in random_lists(num_ex)])

def f_check_if_square_1(num_progs, num_ex=10):
    predicate = lambda x : x in SQUARES
    for i in range(num_progs):
        problem("Check if elements are square",
                [(l, map_predicate(predicate, l)) for l in random_lists(num_ex)])

def f_check_if_three_1(num_progs, num_ex=10):
    predicate = lambda x : x % 3 == 0
    for i in range(num_progs):
        problem("Check if elements are a multiple of three",
                [(l, map_predicate(predicate, l)) for l in random_lists(num_ex)])


#Insert if necessary -- not really needed
# def f_check_if_even_2(num_progs, num_ex=10):
#     predicate = lambda x : x % 2 == 0
#     for i in range(num_progs):
#         problem("Check if elements are even and show inputs",
#                 [(l, map(int, map(predicate, l))) for l in random_lists(num_ex)])

# def f_check_if_odd_2(num_progs, num_ex=10):
#     predicate = lambda x : x % 2 == 1
#     for i in range(num_progs):
#         problem("Check if elements are odd and show inputs",
#                 [(l, map(int, map(predicate, l))) for l in random_lists(num_ex)])

# def f_check_if_prime_2(num_progs, num_ex=10):
#     predicate = lambda x : x in PRIMES
#     for i in range(num_progs):
#         problem("Check if elements are prime and show inputs",
#                 [(l, map(int, map(predicate, l))) for l in random_lists(num_ex)])

# def f_check_if_square_2(num_progs, num_ex=10):
#     predicate = lambda x : x in SQUARES
#     for i in range(num_progs):
#         problem("Check if elements are square and show inputs",
#                 [(l, map(int, map(predicate, l))) for l in random_lists(num_ex)])

# def f_check_if_three_2(num_progs, num_ex=10):
#     predicate = lambda x : x % 3 == 0
#     for i in range(num_progs):
#         problem("Check if elements are a multiple of three and show inputs",
#                 [(l, map(int, map(predicate, l))) for l in random_lists(num_ex)])


# FlashFill-inspired problems
# 0 plays the role of space
def randomWord():
    return [ choice(range(9))+1 for _ in range(choice(range(5))+1) ]
def randomSentence():
    s = []
    l = choice(range(4)) + 1
    for j in range(l):
        if j > 0:
            s += [0]
        s += randomWord()
    return s

# Return the length of a list
def f_first_word(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return the first word',
                [(l,list(takewhile(lambda x: x != 0,l)))
                 for _ in range(num_ex)
                 for l in [randomSentence()] ])

def f_second_word(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return the second word',
                [(p+[0]+l,list(takewhile(lambda x: x != 0,l)))
                 for _ in range(num_ex)
                 for l in [randomSentence()]
                 for p in [randomWord()] ])

def f_third_word(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Return the third word',
                [(p+[0]+l,list(takewhile(lambda x: x != 0,l)))
                 for _ in range(num_ex)
                 for l in [randomSentence()]
                 for p in [randomWord() + [0] + randomWord()] ])

def f_last_word(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Last word',
                [(l,list(reversed(list(takewhile(lambda x: x != 0,reversed(l))))))
                 for _ in range(num_ex)
                 for l in [randomSentence()] ])

def initials(l):
    if l == []: return []
    if l[0] == 0: l = l[1:]
    return [l[0]] + initials(list(dropwhile(lambda x: x != 0,l[1:])))

def bracket(l,b):
    if l[0] != b: l = [b] + l
    if l[-1] != b: l = l + [b]
    return l

def f_get_initials(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Get the initials from the words',
                [(l,initials(l))
                 for _ in range(num_ex)
                 for l in [randomSentence()] ])

def f_duplicate_word(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Duplicate the word',
                [(l,l + [0] + l)
                 for _ in range(num_ex)
                 for l in [randomWord()] ])

def f_drop_word(num_progs, num_ex=10):
    for i in range(num_progs):
        problem('Drop the last word',
                [(l + [0] + w,l)
                 for _ in range(num_ex)
                 for l,w in [(randomSentence(),randomWord())]])

def f_append_word(num_progs, num_ex=10):
    for i in range(num_progs):
        w = randomWord()
        problem('Append constant word %s'%(str(w)),
                [(l,l + [0] + w)
                 for _ in range(num_ex)
                 for l in [randomSentence()] ])

def f_prepend_word(num_progs, num_ex=10):
    for i in range(num_progs):
        w = randomWord()
        problem('Prepend constant word %s'%(str(w)),
                [(l,w + [0] + l)
                 for _ in range(num_ex)
                 for l in [randomSentence()] ])

def f_append_and_prepend_word(num_progs, num_ex=10):
    for i in range(num_progs):
        p,q = randomWord(),randomWord()
        problem('Prepend constant word %s and append constant word %s'%(str(p),str(q)),
                [(l,p + [0] + l + [0] + q)
                 for _ in range(num_ex)
                 for l in [randomSentence()] ])

def f_bracket_input(num_progs, num_ex=10):
    for i in range(num_progs):
        b = i
        problem('Bracket input with delimiter %d' % b,
                [(l,bracket(l,b))
                 for _ in range(num_ex/3)
                 for l in [randomWord() + [b]] ] +
                [(l,bracket(l,b))
                 for _ in range(num_ex/3)
                 for l in [[b] + randomWord()] ] +
                [(l,bracket(l,b))
                 for _ in range(num_ex/3)
                 for l in [randomWord()] ])

def makeTasks():
    # Get all possible programs
    funcs = filter(lambda x : x[:2] == "f_", globals().keys())

    # Total programs per type to generate
    K = 5

    # Total number of examples to generate per program
    E = 10

    for f in funcs:
        strfunc = "%s(%d, %d)" % (f, K, E)
        exec(strfunc)
    return global_problems
