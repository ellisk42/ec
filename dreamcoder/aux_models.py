from dreamcoder.matt.sing import sing
from torch import nn
import torch
from dreamcoder.type import tlist,tbool,tint
from dreamcoder.domains.misc.deepcoderPrimitives import int_to_int, int_to_bool, int_to_int_to_int
from dreamcoder.pnode import PNode,PTask,Examplewise
from dreamcoder.matt.util import *



def new_embedding(H=None):
    if H is None:
        H = sing.cfg.model.H
    return nn.Parameter(torch.randn(1, H))


# TODO I duplicated this from valuehead.py so delete that other copy
class NM(nn.Module):
    def __init__(self, argc, H=None):
        super().__init__()
        if H is None:
            H = sing.cfg.model.H
        self.argc = argc
        if argc > 0:
            self.params = nn.Sequential(nn.Linear(argc*H, H), nn.ReLU(True))
        else:
            self.params = new_embedding(H)
        
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
    def encode_concrete_exwise(self,exwise):
      """
      This gets called by Examplewise.abstract() to encode
      its .concrete field and produce a .abstract field
      """
      sing.stats.call_encode_exwise += 1
      assert exwise.concrete is not None
      return self.encoder.encodeValue(exwise.concrete)

    def encode_ctx(self,ctx):
        """
        Takes a EWContext, abstracts them all, 
        cats on ctx_start and ctx_end vectors, and runs them thru
        the extractor's ctx_encoder GRU.
        """

        # encode everyone
        ctx = [(ew.encode_placeholder(i) if ew.placeholder else ew.get_abstract()) for i,ew in enumerate(ctx)]

        lex = self.encoder.lexicon_embedder
        start = lex(lex.ctx_start).expand(1,sing.num_exs,-1)
        end = lex(lex.ctx_end).expand(1,sing.num_exs,-1)

        ctx = torch.cat([start] + [vec.unsqueeze(0) for vec in ctx] + [end])
        _, res = self.encoder.ctx_encoder(ctx)
        res = res.sum(0) # sum bidir
        return res

    # def encode_known_ctx(self,ctx):
    #   """
    #   Takes a list of Examplewise objects, abstracts them all, 
    #   cats on ctx_start and ctx_end vectors, and runs them thru
    #   the extractor's ctx_encoder GRU.

    #   Note that this doesnt handle the case where there are Nones
    #   in the Examplewise list, hence "known" in the name.

    #   This has the same behavior as inputFeatures from the old days if you pass
    #   in all the inputs to the task as exwise_list
    #   """
    #   sing.stats.call_encode_known_ctx += 1
    #   assert all(c.exwise is not None for c in ctx)

    #   lex = self.encoder.lexicon_embedder
    #   start = lex(lex.ctx_start).expand(1,sing.num_exs,-1)
    #   end = lex(lex.ctx_end).expand(1,sing.num_exs,-1)

    #   ctx = torch.cat([start] + [c.exwise.abstract().unsqueeze(0) for c in ctx] + [end])
    #   _, res = self.encoder.ctx_encoder(ctx)
    #   res = res.sum(0) # sum bidir
    #   return res

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



class ApplyNN(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = sing.cfg
        self.cfg = cfg
        self.apply_cfg = cfg.model.apply


        # self.func_indicator = NM(0)
        # self.out_indicator = NM(0)
        # self.arg_indicators = nn.ModuleList([NM(0) for _ in range(3)])
        self.indicators = nn.ParameterDict({
            'fn': new_embedding(),
            'out': new_embedding(),
            '0': new_embedding(),
            '1': new_embedding(),
            '2': new_embedding(),
            '3': new_embedding(),
        })
        self.get_indicator = lambda label: self.indicators[str(label)]

        self.PREDICT_MASK = new_embedding()

        self.H = cfg.model.H
        self.bindH = {
            'sum': self.H,
            'cat': self.H*2,
        }[self.apply_cfg.bind]

        if self.apply_cfg.type == 'transformer':
            self.readout_from = cfg.model.apply.transformer.readout_from
            self.readout_by =   cfg.model.apply.transformer.readout_by

            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=self.bindH,
                    nhead=cfg.model.apply.transformer.nhead, # vary this
                    dim_feedforward=cfg.model.H*4, # vary this
                    dropout=(0 if sing.cfg.debug.validate_cache else .1)
                ),
                num_layers=cfg.model.apply.transformer.num_layers, # vary this
            )
            if self.readout_by == 'linear':
                self.readout = nn.Linear(self.bindH,cfg.model.H)

        elif self.apply_cfg.type == 'rnn':
            self.gru = nn.GRU(
                input_size=self.bindH, # for indicator+vec
                hidden_size=cfg.model.H,
                num_layers=cfg.model.apply.gru.num_layers,
            )
        else:
            assert False

    def bind(self, label, vec):
        indicator = self.get_indicator(label)
        indicator = indicator.expand(sing.num_exs,-1)
        if self.apply_cfg.bind == 'sum':
            return vec+indicator
        elif self.apply_cfg.bind == 'cat':
            return torch.cat([indicator,vec],dim=-1)
        assert False
    
    def batch_bind(self, labels, vecs):
        """
        take list of labels and vec :: T[BATCH,exs,H]
        returns T[BATCH,exs,bindH]
        (labels get expanded over the examples dimension)
        """
        inds = stack([self.get_indicator(label).expand(sing.num_exs,-1) for label in labels])
        assert inds.shape == vecs.shape
        if self.apply_cfg.bind == 'sum':
            return vecs+inds
        elif self.apply_cfg.bind == 'cat':
            return cat([inds,vecs],dim=-1)
        assert False

    def apply_bound(self,known,target):
        return {
            'rnn': self.gru_apply,
            'transformer': self.transformer_apply,
        }[self.apply_cfg.type](known,target)
        
    def transformer_apply(self,known,target):
        """
        known :: list of [num_exs, bindH] of length ~ argc+1
        target :: bindH   (from bind(target_label,PREDICT))

        returns [num_exs, H] 
        """

        input = rearrange(known+[target], 'seq exs bindH -> seq exs bindH') # stack

        hidden = self.transformer(input) # seq exs bindH -> seq exs bindH

        # readout step 1 to go from 'seq exs bindH -> exs bindH'
        if self.readout_from == 'sum':
            hidden = reduce(hidden, 'seq exs bindH -> exs bindH','sum')
        elif self.readout_from == 'mask': # take the hidden at the last elem of sequence (ie the PREDICT token location)
            hidden = hidden[-1] # seq exs bindH -> exs bindH
        else:
            assert False
        
        # readout step 2 to go from 'exs bindH -> exs H'
        if self.readout_by == 'linear':
            self.apply_cfg.bind == 'cat'
            res = self.readout(hidden)
        elif self.readout_by == 'cut': # cut out at precisely the place that PREDICT_MASK used to be (ie ignore the indicator)
            assert self.apply_cfg.bind == 'cat'
            res = hidden[:,:-self.H]
        elif self.readout_by == 'identity':
            assert self.apply_cfg.bind == 'sum'
            res = hidden # noop
        else:
            assert False

        return res

    def batch_transformer_apply(self,known,target,known_pad_mask):
        """
        known :: 'batch seq exs bindH'
        known_pad_mask :: 'batch seq' says to ignore places in `known` where this mask is 0
        target :: 'batch exs bindH'
        returns :: 'batch exs H'
        """
        # add target on as first element of sequence
        target = rearrange(target, 'batch exs bindH -> batch 1 exs bindH')
        input = cat((target,known),dim=1)
        # also add a corresponding column of `True`s as the first column of the mask
        pad_mask = cat((torch.ones(known.shape[0],1,dtype=bool,device=sing.device),known_pad_mask),dim=1)

        # transformer expects 'seq BATCH H' as shape
        input = rearrange(input, 'batch seq exs bindH -> seq (batch exs) bindH')
        hidden = self.transformer(input, mask=pad_mask, src_key_padding_mask=pad_mask) # seq (batch exs) bindH -> seq (batch exs) H
        hidden = rearrange(hidden, 'seq (batch exs) H -> batch seq exs bindH',exs=sing.num_exs)

        # readout step 1 to go from 'batch seq exs bindH -> batch exs bindH'
        if self.readout_from == 'sum':
            hidden = reduce(hidden, 'batch seq exs bindH -> batch exs bindH','sum')
        elif self.readout_from == 'mask': # take the hidden at the last elem of sequence (ie the PREDICT token location)
            hidden = hidden[:,0,:,:] # batch seq exs bindH -> batch exs bindH
        else:
            assert False
        
        # readout step 2 to go from 'batch exs bindH -> batch exs H'
        if self.readout_by == 'linear':
            self.apply_cfg.bind == 'cat'
            res = self.readout(hidden)
        elif self.readout_by == 'cut': # cut out at precisely the place that PREDICT_MASK used to be (ie ignore the indicator)
            assert self.apply_cfg.bind == 'cat'
            res = hidden[:,:,:-self.H]
        elif self.readout_by == 'identity':
            assert self.apply_cfg.bind == 'sum'
            res = hidden # noop
        else:
            assert False

        return res

    def gru_apply(self,known,target):
        """
        bound_func :: T[num_exs,H]
        bound_args :: list of T[num_exs,H]

        gru wants (argc+1,num_exs,H) as input
        gru gives (argc+1,num_exs,H),(num_layers,num_exs,H) as tuple of outputs. We care about the second tuple (hidden not output) 
        """
        input = rearrange(known+[target], 'seq exs bindH -> seq exs bindH') # stack
        _, hidden = self.gru(input)
        hidden = reduce(hidden, 'seq exs bindH -> exs bindH','sum')
        return hidden


    #def forward(self, func_vec, labelled_arg_vecs=[], parent_vec=None, target_label=None):
    def forward(self, known, target):
        """
        known :: [(label,vec)] where a "label" is 0 | 1 | 2 | 'fn' | 'out'
        target :: label
        """
        target = self.bind(target, self.PREDICT_MASK.expand(sing.num_exs,-1))
        known = [self.bind(label,vec) for label,vec in known]

        res = self.apply_bound(known, target)
        return res

    def batch_forward(self, batch_known, batch_known_labels, batch_target):
        """
        batch_known ::  [BATCH, RAGGED_SEQ, exs, H] with lists as outer two dimensions (and RAGGED_SEQ dim is ragged)
        batch_known_labels ::  [BATCH, RAGGED_SEQ] list of list of labels
        batch_target :: list of labels (len=BATCH)
        """

        # prep target
        mask = repeat(self.PREDICT_MASK, 'H -> batch exs H',batch=len(batch_target),exs=sing.num_exs)
        # mask = self.PREDICT_MASK.expand(len(batch_target),sing.num_exs,-1) # [BATCH, exs, H]
        batch_target = self.batch_bind(batch_target, mask) # -> 'batch exs bindH'

        # prep known
        batch_known,pad_mask = pad_list_list_tensor(batch_known) # batch_known :: [BATCH, SEQ, exs, H]
        # pad the labels to [BATCH, SEQ] list of lists
        longest_seq = pad_mask.shape[1]
        batch_known_labels = [labels + ['out']*(longest_seq-len(labels)) for labels in batch_known_labels] # 'out' is a garbage label here just for padding
        # flatten so batch_bind can bind over the sequence and batch dimension both at once
        batch_known_labels = flatten(batch_known_labels)
        batch_known = rearrange(batch_known,'batch seq exs H -> (batch seq) exs H')
        # batch bind
        batch_known = self.batch_bind(batch_known,batch_known_labels)
        # unflatten
        batch_known = rearrange(batch_known,'(batch seq) exs bindH -> batch seq exs bindH',seq=longest_seq)

        assert sing.cfg.apply.type == 'transformer', "gru not yet implemented for batching"
        return self.batch_transformer_apply(batch_known, batch_target, pad_mask)
        


def distance(target_vec, cand_vec):
    raise NotImplementedError # dot prod

def batch_distance(target_vec, cand_vecs):
    raise NotImplementedError # mat*vec mul
