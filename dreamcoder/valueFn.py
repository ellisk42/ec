#value fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def serialize_and_flatten(prog):
    flattened_seq = []
    for subtree in prog.serialize():
        if type(subtree) == str:
            flattened_seq.append(subtree)
        else:
            flattened_seq.extend( serialize_and_flatten(subtree) )
    return flattened_seq

def stringify(line):
    lst = []
    string = ""       
    for char in line+" ": 
        if char == " ":  
            if string != "":      
                lst.append(string)     
            string = ""              
        elif char in '()':            
            if string != "":        
                lst.append(string)     
            string = ""                
            lst.append(char)        
        else:                       
            string += char       
    return lst

class RNNSketchEncoder(nn.Module):
    """
    Encodes a program with an RNN
    todo: 
    [ ] CUDA!!
    [X] DSL
    """

    def __init__(self, g, H=512):
        #specEncoder can be None, meaning you dont use the spec at all to encode objects
        super(RNNSketchEncoder, self).__init__()
        self.use_cuda = False #FIX THIS

        extras = ['(', ')', 'lambda'] + ['$'+str(i) for i in range(10)] 

        self.lexicon = [str(p) for p in g.primitives] + extras
        self.embedding = nn.Embedding(len(self.lexicon), H)

        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }

        self.model = nn.GRU(H,H,1)
        self.H = H
        self.outputDimensionality = H


    def forward(self, sketches):
        #don't use spec, just there for the API
        assert type(sketches) == list

        #idk if obj is a list of objs... presuably it ususaly is 
        tokens_list = [ stringify(str(sketch)) for sketch in sketches]

        symbolSequence_list = [[self.wordToIndex[t] for t in tokens] for tokens in tokens_list]

        inputSequences = [torch.tensor(ss) for ss in symbolSequence_list] #this is impossible

        if self.use_cuda: #TODO
            inputSequences = [s.cuda() for s in inputSequences]

        inputSequences = [self.embedding(ss) for ss in inputSequences]

        # import pdb; pdb.set_trace()
        idxs, inputSequence = zip(*sorted(enumerate(inputSequences), key=lambda x: -len(x[1])  ) )
        try:
            packed_inputSequence = torch.nn.utils.rnn.pack_sequence(inputSequence)
        except ValueError:
            print("padding issues, not in correct order")
            import pdb; pdb.set_trace()

        _,h = self.model(packed_inputSequence) #dims
        unperm_idx, _ = zip(*sorted(enumerate(idxs), key = lambda x: x[1]))
        h = h[:, unperm_idx, :]
        h = h.squeeze(0)
        #o = o.squeeze(1)

        objectEncodings = h
        return objectEncodings






class NMN(nn.Module):
    def __init__(self, operator, H=128):
        super(NMN, self).__init__()
        self.operator = operator
    

        if operator.type.isArrow:
            n_args = len(operator.type.arguments)
            self.n_args = n_args
            #can just do a stack I think ...
            if n_args == 1:
                self.params = nn.Sequential(nn.Linear(H, H), nn.ReLU())

            elif n_args == 2:
                self.params = nn.Sequential(nn.Linear(2*H, H), nn.ReLU())
            else:
                assert False, "more than two inputs not supported"

        else:
            self.n_args = 0
            self.params = nn.Parameter(torch.randn(H))
        



    def forward(self, *args):
        if self.n_args == 0:
            return self.params
        else: 
            inp = torch.cat(args, 0) #TODO i think this is correct dim
            out = self.params(inp)
            return out

        #TODO .. api of this


class ModularSketchEncoder(nn.Module):
    """
    ModuleEncoder, unbatched for now
    """
    def __init__(self, g, H=512):
        super(NMObjectEncoder, self).__init__()
        #self.initialState = nn.Linear(self.specEncoder.outputDimensionality, H)

        self.lexicon = g + []

        self.fn_modules = nn.ModuleDict()

        for op in DSL.operators:
            self.fn_modules[op.token] = NMN(op, H)


        #self.embedding = nn.Embedding(len(self.lexicon), H)
        #self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
        #self.model = nn.GRU(H,H,1)
        self.H = H
        self.outputDimensionality = H

    def forward(self, sketch):
        """
        spec is a list of specs
        obj is a list of objs

        basic idea: slurp up obj encoding. for now probably don't use spec??
        unbatched for now
        """

        if sketch.isApplication:

            f, xs = sketch.applicationParse()

            return self(f)(*map(self.forward, xs))

        elif sketch.isPrimitive or sketch.isInvented:
            return self.fn_modules[sketch]

        elif sketch.isHole:
            return self.holeModules #TODO need for each type i think


        elif sketch.isAbstraction:
            return self. sketch.body


        elif sketch.isIndex:
            #nasty ...
            pass

        ### old code
        def apply(p):
            #recursive helper fn
            if not p.type.isArrow:
                return self.fn_modules[p.token]()
            else:
                return self.fn_modules[p.token](*map(apply, p.children()))

        out = []
        for o in obj: #this is a batch, I believe
            # ideally should memoize so I'm not redoing work here
            out.append( apply(o) )
        
        return torch.stack(out, dim=0)



if __name__ == '__main__':
    try:
        import binutil  # required to import from dreamcoder modules
    except ModuleNotFoundError:
        import bin.binutil  # alt import if called as module



    from dreamcoder.domains.arithmetic.arithmeticPrimitives import *
    g = ContextualGrammar.fromGrammar(Grammar.uniform([k0,k1,addition, subtraction]))
    g = g.randomWeights(lambda *a: random.random())
    #p = Program.parse("(lambda (+ 1 $0))")

    m = RNNSketchEncoder(g)

    request = arrow(tint,tint)
    for ll,_,p in g.enumeration(Context.EMPTY,[],request,
                               12.):
        ll_ = g.logLikelihood(request,p)
        print(ll,p,ll_)
        d = abs(ll - ll_)
        assert d < 0.0001

        encoding = m([p])

        assert 0



