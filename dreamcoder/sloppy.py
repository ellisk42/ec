# a sloppy approach to observational equivalence
# runs free variables on random inputs
import random
import itertools

from dreamcoder.frontier import *
from dreamcoder.program import *
from dreamcoder.type import *
from dreamcoder.utilities import *
import dreamcoder as dc


class Sloppy():
    def __init__(self, inputs, n=6, sound=True, continuationType=None,
                 request=None):
        self.continuationType=continuationType
        self.sound = sound
        self.inputs = inputs
        self.next_symbol = 0
        self._test_inputs = {}
        self.n=n

        if continuationType is not None:
            assert all( len(xs)==1 for xs in inputs ), "for no good reason we require continuation types to only have one argument. ask Kevin to fix this later"
            self.continuation_values = [ [ k
                                           for k, t in zip(input_data, request.functionArguments())
                                           if t==continuationType][0]
                                        for input_data in self.inputs ]

    def unique_symbol(self):
        self.next_symbol += 1
        return self.next_symbol    

    def sound_signature(self, expression, tp, arguments):
        if self.inputs is None: 
            raise Exception("No example inputs provided in Sloppy!")

        # eprint()
        # eprint(expression, tp, arguments)
        # for i in expression.freeVariables():
        #     eprint("free ", i, arguments[i])
            
        outputs = []
        for i in expression.freeVariables():
            #illegal?
            if self.continuationType is None and i < len(arguments) - len(self.inputs[0]) or\
               (self.continuationType is not None and self.continuationType!=arguments[i]):
                #eprint("invalid")
                return self.unique_symbol()
        #eprint("good work")

        if tuple(arguments) not in self._test_inputs:
            environments = []

            for ti, test_input in enumerate(self.inputs):
                if self.continuationType is None:
                    environment = [None]*(len(arguments) - len(test_input))+list(reversed(test_input))
                if self.continuationType is not None:
                    environment = [None]*len(arguments)

                    for i in range(len(arguments)):
                        if arguments[i] == self.continuationType:
                            environment[i] = self.continuation_values[ti]

                environments.append(environment)
            self._test_inputs[tuple(arguments)] = environments
        else:
            environments = self._test_inputs[tuple(arguments)]

        
        for environment in environments:
            try:
                o = expression.evaluate(environment)
            except Exception as e:
                o = None
            if o is None:
                outputs.append(None)
                continue
            try:
                outputs.append(self.value_to_key(o, tp))
                hash(outputs[-1])
            except:
                eprint(expression, tp, environment, o, test_input)
                assert False
        #eprint("output", tuple(outputs))
        if all(o is None for o in outputs):
            return None
        return tuple(outputs)

    def compute_signature(self, expression, tp, arguments):
        
        if self.sound: return self.sound_signature(expression, tp, arguments)
        
        outputs = []
        for test_input in self.test_inputs(arguments):
            try:
                o = expression.evaluate(test_input)
            except:
                o = None
            if o is None:
                outputs.append(None)
                continue
            try:
                outputs.append(self.value_to_key(o, tp))
            except:
                eprint(expression, tp, o, test_input)
                assert False
        if all(o is None for o in outputs):
            return None
        return tuple(outputs)

    def possible_values(self, tp):
        if tp==self.continuationType:
            return ["CONTINUATION"]
        if str(tp) == "int":
            return random.choices(range(-10,0), k=10//2-2) +\
                random.choices(range(2, 10), k=10//2-1) +\
                [-1,0,1]
        if str(tp) == "bool":
            return [False,True]
        if str(tp) == "real":
            return [random.random()*10-5 for _ in range(self.n) ]
        if str(tp) == "tsize":
            return [4]
        if isinstance(tp, TypeConstructor):
            if tp.name=="list":
                return [ [ random.choice(self.possible_values(tp.arguments[0]))
                           for _ in range(random.choice(range(4))) ]
                         for _ in range(self.n-1) ]+\
                             [[]]
            if tp.isArrow:
                assert False, "not supported function types for observational equivalents"
        assert False, f"unsupported type {tp}"
        

    def test_inputs(self, arguments):
        if tuple(arguments) in self._test_inputs:
            return self._test_inputs[tuple(arguments)]

        if self.inputs is not None and self.continuationType is None:
            # drop the last arguments because they correspond to the inputs
            number_of_arguments = len(self.inputs[0])
            arguments = arguments[:-number_of_arguments]

        test_inputs=[]
        for sloppy in itertools.product(*(self.possible_values(a) for a in arguments)):
            if self.inputs is not None:
                for ti, input_tuple in enumerate(self.inputs):
                    if self.continuationType is None:
                        test_inputs.append(list(sloppy) + list(reversed(input_tuple)))
                    else:
                        # insert the continuation objects
                        test_inputs.append(\
                                    [self.continuation_values[ti] if xx=="CONTINUATION" else xx
                                     for xx in sloppy ])               
            else:
                test_inputs.append(list(sloppy))
        test_inputs = random.sample(test_inputs, min(max(2, len(self.inputs)),
                                                     len(test_inputs)))

        
        self._test_inputs[tuple(arguments)] = test_inputs
        return test_inputs
    
    def value_to_key(self, value, output_type):
        if output_type == dc.domains.quantum_circuits.primitives.tcircuit:
            # Here we only need check the unitary
            unitary = dc.domains.quantum_circuits.primitives.circuit_to_mat(value)
            value = unitary.tobytes()
        elif str(output_type)=="tower":
            state, plan = value(dc.domains.tower.towerPrimitives.TowerState())
            value = (state.hand, state.orientation, tuple(plan))
        elif isinstance(output_type, TypeConstructor) and output_type.name=="list":
            value = tuple(self.value_to_key(v, output_type.arguments[0])
                          for v in value )
        return value

    
