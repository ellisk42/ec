from dreamcoder.matt.sing import sing
from torch import nn
import torch
from dreamcoder.type import tlist,tbool,tint
from dreamcoder.domains.misc.deepcoderPrimitives import int_to_int, int_to_bool, int_to_int_to_int
from dreamcoder.pnode import PNode,PTask,Examplewise

# TODO I duplicated this from valuehead.py so delete that other copy
class NM(nn.Module):
    def __init__(self, argc, H=None):
        super().__init__()
        if H is None:
            H = sing.cfg.model.H
        self.argc = argc
        if argc > 0:
            self.params = nn.Sequential(nn.Linear(argc*H, H), nn.ReLU())
        else:
            self.params = nn.Parameter(torch.randn(1, H))
        
    def forward(self, *args):
        assert len(args) == self.argc
        if self.argc == 0:
            return self.params
        args = torch.cat(args,dim=1) # cat along example dimension. Harmless if only one thing in args anyways
        return self.params(args)

class ProgramRNN(nn.Module):
    def __init__(self):
      super().__init__()
      H = self.H = sing.cfg.model.H
      extras = ['(', ')', 'lambda', '<HOLE>', '#'] + ['$'+str(i) for i in range(15)] 
      self.lexicon = [str(p) for p in sing.g.primitives] + extras
      self.embedding = nn.Embedding(len(self.lexicon), H)
      self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
      self.model = nn.GRU(H,H,1)
    def encode_sketches(self, sketch_strs):
        #don't use spec, just there for the API
        # assert type(sketches) == list
        #idk if obj is a list of objs... presuably it ususaly is 
        from dreamcoder.valueHead import stringify
        # tokens_list = [ stringify(str(sketch)) for sketch in sketches]
        tokens_list = [ stringify(sketch) for sketch in sketch_strs]
        symbolSequence_list = [[self.wordToIndex[t] for t in tokens] for tokens in tokens_list]
        inputSequences = [torch.tensor(ss,device=sing.device) for ss in symbolSequence_list] #this is impossible
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

class AbstractionFn(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      from dreamcoder.domains.list.main import ListFeatureExtractor
      self.encoder = ListFeatureExtractor(maximumLength=sing.taskloader.L_big)
    def encode_exwise(self,exwise):
      """
      This gets called by Examplewise.abstract() to encode
      its .concrete field and produce a .abstract field
      """
      assert exwise.concrete is not None
      sing.stats.call_encode_exwise += 1
      return self.encoder.encodeValue(exwise.concrete)

    def encode_known_ctx(self,ctx):
      """
      Takes a list of Examplewise objects, abstracts them all, 
      cats on ctx_start and ctx_end vectors, and runs them thru
      the extractor's ctx_encoder GRU.

      Note that this doesnt handle the case where there are Nones
      in the Examplewise list, hence "known" in the name.

      This has the same behavior as inputFeatures from the old days if you pass
      in all the inputs to the task as exwise_list
      """
      sing.stats.call_encode_known_ctx += 1
      assert all(c.exwise is not None for c in ctx)

      lex = self.encoder.lexicon_embedder
      start = lex(lex.ctx_start).expand(1,sing.num_exs,-1)
      end = lex(lex.ctx_end).expand(1,sing.num_exs,-1)

      ctx = torch.cat([start] + [c.exwise.abstract().unsqueeze(0) for c in ctx] + [end])
      _, res = self.encoder.ctx_encoder(ctx)
      res = res.sum(0) # sum bidir
      return res

class AbstractTransformers(nn.Module):
  def __init__(self):
    super().__init__()
    H = sing.cfg.model.H
    # hole NMs
    possible_hole_tps = [tint,tbool,tlist(tint)]
    if not sing.cfg.data.expressive_lambdas:
        possible_hole_tps += [int_to_int, int_to_bool, int_to_int_to_int]
    self.hole_nms = nn.ModuleDict()
    for tp in possible_hole_tps:
        self.hole_nms[tp.show(True)] = NM(1)

    # populate fnModules
    self.fn_nms = nn.ModuleDict()
    for p in sing.g.primitives:
        argc = len(p.tp.functionArguments())
        self.fn_nms[p.name] = nn.ModuleList([NM(argc) for _ in range(argc+1)])

    self.index_nm = NM(0)
    # TODO in the future to allow for >1 toplevel arg the above could be replaced with:
    # self.index_nms = [NM(0,cfg.model.H) for _ in range(max_index+1)]

    # TODO this is kept the same as the BAS paper however is totally worth experimenting with in the future
    # (we'd like to improve how lambdas get encoded)
    nargs = 1 if sing.cfg.model.pnode.ctxful_lambdas else 0
    self.lambda_index_nms = nn.ModuleList([NM(nargs) for _ in range(2)])
    self.lambda_hole_nms = nn.ModuleDict()
    for tp in [tint,tbool]:
        self.lambda_hole_nms[tp.show(True)] = NM(nargs)

class AbstractComparer(nn.Module):
  def __init__(self):
      super().__init__()
      H = sing.cfg.model.H
      self.compare_module = nn.Sequential(nn.Linear(H*2, H), nn.ReLU())

  def forward(self, sk_reps, output_feats):
    compare_input = torch.cat((sk_reps,output_feats),dim=2) # [num_sketches,num_exs,H*2]
    compared = self.compare_module(compare_input) # [num_sketches,num_exs,H]
    return compared
