import random
from collections import defaultdict
import json
import math
import os
import datetime
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import itertools
import mlb

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.list.listPrimitives import basePrimitives, primitives, McCarthyPrimitives, bootstrapTarget_extra, no_length
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.list.makeListTasks import make_list_bootstrap_tasks, sortBootstrap, EASYLISTTASKS

from dreamcoder.domains.misc.deepcoderPrimitives import deepcoderPrimitives
#from dreamcoder.domains.list.makeDeepcoderData import DeepcoderTaskloader




class ExtractorGenerator:
    def __init__(self,cfg,maximumLength):
        self.cfg = cfg
        self.maximumLength = maximumLength
        self._groups = {}
    def __call__(self, group):
        """
        Returns an extractor object. If called twice with the same group (an int or string or anything) the same object will be returned (ie share weights)
        """
        if group not in self._groups:
            self._groups[group] = ListFeatureExtractor(maximumLength=self.maximumLength, cfg=self.cfg)
        return self._groups[group]


def retrieveJSONTasks(filename, features=False):
    """
    For JSON of the form:
        {"name": str,
         "type": {"input" : bool|int|list-of-bool|list-of-int,
                  "output": bool|int|list-of-bool|list-of-int},
         "examples": [{"i": data, "o": data}]}
    """
    with open(filename, "r") as f:
        loaded = json.load(f)
    TP = {
        "bool": tbool,
        "int": tint,
        "list-of-bool": tlist(tbool),
        "list-of-int": tlist(tint),
    }
    return [Task(
        item["name"],
        arrow(TP[item["type"]["input"]], TP[item["type"]["output"]]),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
        features=(None if not features else list_features(
            [((ex["i"],), ex["o"]) for ex in item["examples"]])),
        cache=False,
    ) for item in loaded]


def list_features(examples):
    if any(isinstance(i, int) for (i,), _ in examples):
        # obtain features for number inputs as list of numbers
        examples = [(([i],), o) for (i,), o in examples]
    elif any(not isinstance(i, list) for (i,), _ in examples):
        # can't handle non-lists
        return []
    elif any(isinstance(x, list) for (xs,), _ in examples for x in xs):
        # nested lists are hard to extract features for, so we'll
        # obtain features as if flattened
        examples = [(([x for xs in ys for x in xs],), o)
                    for (ys,), o in examples]

    # assume all tasks have the same number of examples
    # and all inputs are lists
    features = []
    ot = type(examples[0][1])

    def mean(l): return 0 if not l else sum(l) / len(l)
    imean = [mean(i) for (i,), o in examples]
    ivar = [sum((v - imean[idx])**2
                for v in examples[idx][0][0])
            for idx in range(len(examples))]

    # DISABLED length of each input and output
    # total difference between length of input and output
    # DISABLED normalized count of numbers in input but not in output
    # total normalized count of numbers in input but not in output
    # total difference between means of input and output
    # total difference between variances of input and output
    # output type (-1=bool, 0=int, 1=list)
    # DISABLED outputs if integers, else -1s
    # DISABLED outputs if bools (-1/1), else 0s
    if ot == list:  # lists of ints or bools
        omean = [mean(o) for (i,), o in examples]
        ovar = [sum((v - omean[idx])**2
                    for v in examples[idx][1])
                for idx in range(len(examples))]

        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
        cnt_not_in_output = [cntr(i, o) for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [len(o) for (i,), o in examples]
        features.append(sum(len(i) - len(o) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(om - im for im, om in zip(imean, omean)))
        features.append(sum(ov - iv for iv, ov in zip(ivar, ovar)))
        features.append(1)
        # features += [-1 for _ in examples]
        # features += [0 for _ in examples]
    elif ot == bool:
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [-1 for _ in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += [0 for _ in examples]
        features.append(0)
        features.append(sum(imean))
        features.append(sum(ivar))
        features.append(-1)
        # features += [-1 for _ in examples]
        # features += [1 if o else -1 for o in outs]
    else:  # int
        def cntr(
            l, o): return 0 if not l else len(
            set(l).difference(
                set(o))) / len(l)
        cnt_not_in_output = [cntr(i, [o]) for (i,), o in examples]
        outs = [o for (i,), o in examples]

        #features += [len(i) for (i,), o in examples]
        #features += [1 for (i,), o in examples]
        features.append(sum(len(i) for (i,), o in examples))
        #features += cnt_not_int_output
        features.append(sum(cnt_not_in_output))
        features.append(sum(o - im for im, o in zip(imean, outs)))
        features.append(sum(ivar))
        features.append(0)
        # features += outs
        # features += [0 for _ in examples]

    return features


def isListFunction(tp):
    try:
        Context().unify(tp, arrow(tlist(tint), t0))
        return True
    except UnificationFailure:
        return False


def isIntFunction(tp):
    try:
        Context().unify(tp, arrow(tint, t0))
        return True
    except UnificationFailure:
        return False


class Lexicon(nn.Embedding):
    """
    roughly use like:
        idxs_list = idxs_of_toks(tok_list)
        idxs
    """
    def __init__(self, lexicon, H=64, cuda=True, cfg=None, *args, **kwargs):
        if cfg is not None:
            cuda = cfg.cuda
            H = cfg.model.H
        else:
            print(f'warning: {self.__class__.__name__} initialized with no `cfg` (was this intentional?)')
        self.lexicon = lexicon.union({'<UNK>','<PAD>'})
        self.length = len(self.lexicon)
        self.idx_of_tok = {tok:torch.tensor(idx,dtype=torch.long) for idx,tok in enumerate(self.lexicon)}
        if cuda:
            self.idx_of_tok = {k:v.cuda() for k,v in self.idx_of_tok.items()}
        self.pad = self.idx_of_tok['<PAD>']
        self.unk = self.idx_of_tok['<UNK>']
        self.H = H
        self.use_cuda = cuda
        super().__init__(num_embeddings=self.length, embedding_dim=H, padding_idx=int(self.pad), *args, **kwargs)
        """
        actual number of embeddings will be length+2 where an extra one is added
        """
    def idxs_of_toks(self,tok_list):
        """
        works on arbitrarily many nested lists as well
        [tok] -> [longtensor]
        """
        res = []
        # sanitize
        unk = 0
        for tok in tok_list:
            if isinstance(tok,(list,tuple)):
                res.append(self.idxs_of_toks(tok))
            elif isinstance(tok,bool):
                res.append(self.idx_of_tok['<'+str(tok)+'>']) # True -> "<True>". Needed bc dictionaries can't tell the difference between 1 and True
            elif tok not in self.lexicon:
                if unk == 0:
                    mlb.yellow(f"converting {tok} to <UNK>. suppressing future warnings")
                unk += 1
                res.append(self.unk)
            else:
                res.append(self.idx_of_tok[tok])
        if unk > 0:
            mlb.yellow(f"Total <UNK>s: {unk}")
        # convert to indices for embeddings
        return res
    def pad_idxs_lists(self,idxs_lists):
        assert isinstance(idxs_lists,list)
        assert isinstance(idxs_lists[0],list)
        sizes = [len(idxs_list) for idxs_list in idxs_lists]
        pad_till = max(sizes)

        # padding
        for i, idxs_list in enumerate(idxs_lists):
            idxs_lists[i] += [self.pad] * (pad_till - len(idxs_list))
        return idxs_lists, torch.tensor(sizes)
    def sort_and_pack(self,embeddings,sizes):
        mlb.log(f'using these sizes for gru: {sizes.tolist()}')
        assert sizes.max() == embeddings.shape[1]
        assert embeddings.dim() == 3
        sizes,sorter = sizes.sort(descending=True)
        _,unsorter = sorter.sort() # fun trick
        mlb.log(f'sorting embeddings by sorter: {sorter.tolist()}')
        embeddings = embeddings[sorter] # sort by decreasing size
        embeddings = embeddings.permute(1,0,2) # swap first two dims. [padded_ex_length, num_exs, H]
        mlb.log(f'permuted to {tuple(embeddings.shape)} :: (padded_length, batch_size, H)')
        packed = pack_padded_sequence(embeddings,sizes)
        return packed,unsorter


class ListFeatureExtractor(RecurrentFeatureExtractor):
    special = None
    def __init__(self, maximumLength, cfg):
        c = cfg.model.extractor
        modular = c.modular
        bidir_ctx = c.bidir_ctx
        bidir_int = c.bidir_int
        digitwise = c.digitwise
        cuda = cfg.cuda
        H = cfg.model.H

        self.lexicon = {"LIST_START", "LIST_END", "INT_START", "INT_END", 'CTX_START', 'CTX_END', "?", "<True>", "<False>"}
        if digitwise:
            self.lexicon = self.lexicon | set(map(str,range(0,10))) | {'-'}
        else:
            self.lexicon = self.lexicon | set(range(-64,65))
        # add all non-arrow primitives ie things that should count as concrete values as opposed to arrows that are represented with NMs
        self.lexicon = self.lexicon | {p.value for p in deepcoderPrimitives() if not p.tp.isArrow()}

        self.maximumLength = maximumLength

        self.modular = modular
        self.digitwise = digitwise
        if digitwise:
            assert modular

        #self.recomputeTasks = True

        super().__init__(
            lexicon=list(self.lexicon),
            tasks=[],
            cuda=cuda,
            H=H,
            bidirectional=True)

        self.lexicon = set(self.lexicon)

        # encodes a [int]
        if self.modular:
            self.list_encoder = nn.GRU(input_size=H, hidden_size=H, num_layers=1, bidirectional=True)
            self.ctx_encoder = nn.GRU(input_size=H, hidden_size=H, num_layers=1, bidirectional=bidir_ctx)
        if self.digitwise:
            self.int_encoder = nn.GRU(input_size=H, hidden_size=H, num_layers=1, bidirectional=bidir_int)
            self.digit_embedder = Lexicon(self.lexicon, cfg=cfg)
    @property # must be a property so that it gets updated when we get map_locationed somewhere else
    def device(self):
        return self.list_encoder.all_weights[0][0].device


    def tokenize(self, examples):
        def sanitize(l): 
            ret = []
            for z_ in l:
                for z in (z_ if isinstance(z_, list) else [z_]):
                    if (z in self.lexicon) or isinstance(z, torch.Tensor):
                        ret.append(z)
                    else:
                        #print (f"WARNING: the following token was not tokenized: {z}")
                        ret.append("?")
            return ret

        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            y = sanitize(y)
            if len(y) > self.maximumLength:
                return None

            serializedInputs = []
            for xi, x in enumerate(xs):
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                x = sanitize(x)
                if len(x) > self.maximumLength:
                    return None
                serializedInputs.append(x)

            tokenized.append((tuple(serializedInputs), y))

        return tokenized
    def inputFeatures(self,task):
        if not self.modular:
            # TODO old code just kept for comparison
            tokenized = self.tokenize(task.examples)
            assert tokenized

            # this deletes any LIST_START LIST_END inserted by .tokenize()
            tokenized = [(ex[0],[]) for ex in tokenized]

            e = self.examplesEncoding(tokenized)
            return e
        mlb.log('in inputFeatures()')
        inputs = [ex[0] for ex in task.examples] # [num_exs,num_args]
        argwise = list(zip(*inputs)) # [num_args,num_exs]
        mlb.log('inputs (argwise):')
        for i,arg in enumerate(argwise):
            mlb.log(f'\targ {i}:')
            for j,ex in enumerate(arg):
                mlb.log(f'\t\tex {j}: {ex}')
        
        inputs = torch.stack([self.encodeValue(arg) for arg in argwise]) # [num_args,num_exs,H]
        mlb.log(f'stacked arg encodings :: {tuple(inputs.shape)} :: (num_args,5,H) :: (seq_length, batch, H) which is correct for non-packed GRU')
        mlb.log('running ctx_encoder() gru on stacked results of encodeValue')
        ctx_start_vec = self.digit_embedder(self.digit_embedder.idx_of_tok['CTX_START']).expand(1,len(task.examples),-1)
        ctx_end_vec = self.digit_embedder(self.digit_embedder.idx_of_tok['CTX_END']).expand(1,len(task.examples),-1)
        inputs = torch.cat((ctx_start_vec,inputs,ctx_end_vec))
        mlb.log('catted on ctx_start and ctx_end vectors')
        mlb.log(f'input to ctx_encoder: {tuple(inputs.shape)}')
        _, res = self.ctx_encoder(inputs)
        res = res.sum(0) # sum over bidirectionality (if any) [num_exs,H]
        mlb.log(f'inputFeatures returning {tuple(res.shape)}')
        return res
    def run_tests(self):
        atol = 1e-7
        close = lambda t1, t2: t1.allclose(t2,atol=atol)
        x = [[1,2,3,4],[1,2,3,4]]
        res = self.encodeValue(x)
        assert close(res[0],res[1]), "encodeValue is not acting independently on each example"

        x1 = [[1,2,3,4],[1,1,1,1]]
        x2 = [[1,2,3,4],[2,2,3,6]] # vary the 2nd example
        x3 = [[1,2,3,4],[2,2]] # vary length of 2nd example
        res1 = self.encodeValue(x1)
        res2 = self.encodeValue(x2)
        res3 = self.encodeValue(x3)
        assert close(res1[0],res2[0]), "contents of 2nd example is affecting 1st"
        assert close(res1[0],res3[0]), "length of 2nd example is affecting 1st"
        assert not close(res1[0],res1[1]), "your allclose() metric is not good for telling if things are equal"

        x = [1,1]
        res = self.encodeValue(x)
        assert close(res[0],res[1])

        x1 = [1,2]
        x2 = [1,3] # vary the 2nd example
        res1 = self.encodeValue(x1)
        res2 = self.encodeValue(x2)
        assert close(res1[0],res2[0]), "contents of 2nd example is affecting 1st"
        assert not close(res1[0],res1[1]), "your allclose() metric is not good for telling if things are equal"

        return
    def outputFeatures(self,task):
        if not self.modular:
            # TODO old code just kept for comparison
            tokenized = self.tokenize(task.examples)
            assert tokenized

            # this deletes any LIST_START LIST_END inserted by .tokenize()
            tokenized = [([],ex[1]) for ex in tokenized]

            e = self.examplesEncoding(tokenized)
            return e
        outputs = [ex[1] for ex in task.examples] # [num_exs]
        mlb.log('in outputFeatures()')
        mlb.log('outputs:')
        for i,output in enumerate(outputs):
            mlb.log(f'\tex {i}: {output}')
        mlb.log('running encodeValue() on outputs')
        res = self.encodeValue(outputs)
        mlb.log('outputFeatures() is returning')
        return res
        
    def tokensToIndices(self,token_list):
        assert not self.digitwise
        # sanitize
        unk = 0
        for i,token in enumerate(token_list):
            if isinstance(token,bool):
                token = '<'+str(token)+'>' # True -> "<True>". Needed bc dictionaries can't tell the difference between 1 and True
                token_list[i] = token
            if token not in self.symbolToIndex:
                if unk == 0:
                    #assert False
                    mlb.yellow(f"converting {token} to <UNK>. suppressing future warnings")
                unk += 1
                token_list[i] = "?"
        if unk > 0:
            mlb.yellow(f"Total <UNK>s: {unk}")
        # convert to indices for embeddings
        return [self.symbolToIndex[token] for token in token_list]

    def pad_indices_lists(self,indices_lists):
        # TODO note this is not used in digitwise i think
        assert not self.digitwise
        assert isinstance(indices_lists,list)
        assert isinstance(indices_lists[0],list)
        sizes = [len(indices_list) for indices_list in indices_lists]
        pad_till = max(sizes)

        # padding
        for i, indices_list in enumerate(indices_lists):
            indices_lists[i] += [self.endingIndex] * (pad_till - len(indices_list))
        return indices_lists, torch.tensor(sizes)

    def encodeValue(self, val):
        """
        val :: a concrete value that shows up as an intermediate value
            during computation and converts it to a vector. Note that this is
            always a list of N such values sine we're evaluating on all N examples
            at once. val :: [int] or [bool] or [[int]] where the outer list
            is always the list of examples.
        returns :: Tensor[num_exs,H]
        """
        mlb.log('in encodeValue()')
        is_list = lambda v: isinstance(v,(list,tuple))
        is_int = lambda v: isinstance(v,(int,np.integer))
        is_bool = lambda v: isinstance(v,(bool,np.bool_))

        assert is_list(val)
        #assert len(val) == self.deepcoder_taskloader.N
        if is_list(val[0]):
            if not self.modular:
                # TODO old code, only keeping it around for comparison
                assert len(val[0]) == 0 or is_int(val[0][0]) # make sure its [[int]]
                class FakeTask: pass
                fake_task = FakeTask()
                fake_task.examples = [((v,),[]) for v in val] # :: [(([Int],),[])] in the single argument case
                vec = self.inputFeatures(fake_task) # [5,H]
                return vec

            if len(val[0]) > 0 and is_bool(val[0][0]):
                # if this happens you wanna go back and add true/false to lexicon but you can't do that directly, you need to 
                # use the "True" and "False" strings (to avoid clash with 0/1 hashes) 
                # and then here in encodeValue you can map a translation from bool to string "True"/"False" over the lists
                raise NotImplementedError


            if self.digitwise:
                exs = val # :: [num_exs,num_ints] - num_ints is number of ints in an example list
                exs = [[['INT_START']+list(str(_int))+['INT_END'] for _int in ex] for ex in exs] # convert 100 -> ['1','0','0']
                # exs :: [num_exs,num_ints,num_digits] = num_digits is number of digits in an int
                before_idxs_of_toks = exs # in case you want to inspect during debugging
                mlb.log('digitified examples:')
                for i,ex in enumerate(before_idxs_of_toks):
                    mlb.log(f'\tex {i}:')
                    for _int in ex:
                        mlb.log(f'\t\t{_int}')
                exs = self.digit_embedder.idxs_of_toks(exs) # ::[[[longtensor]]] (num_exs,list_len,num_digits)

                if mlb.get_verbose():
                    mlb.log('to indices:')
                    for i,ex in enumerate(exs):
                        mlb.log(f'\tex {i}:')
                        for _int in ex:
                            mlb.log(f'\t\t{list(map(int,_int))}')

                ints_per_ex = [len(l) for l in exs]
                # flatten over examples (so its one massive list of ints each represented as a list of digits)
                ints = list(itertools.chain.from_iterable(exs)) # ::[[longtensor]] (num_exs*list_len, num_digits)
                if len(ints) > 0:
                    # pad it to a constant number of digits per int
                    ints, digits_per_int = self.digit_embedder.pad_idxs_lists(ints)
                    ints = torch.stack([torch.stack(digits) for digits in ints]) # :: [num_exs*list_len, longest_num_digits]
                    mlb.log('flattened, padded, stacked:')
                    mlb.log(ints)
                    mlb.log(f'shape:{tuple(ints.shape)} :: (ints, longest_num_digits)')
                    mlb.log('running pointwise digit_embedder()')
                    ints_embedded = self.digit_embedder(ints) # pointwise, converts indices -> embeddings. [num_exs*list_len, longest_num_digits, H]
                    mlb.log(f'ints_embedded :: {tuple(ints_embedded.shape)} :: (ints, longest_num_digits, H)')
                    mlb.log(f'Heres the ints_embedded[0] which is the digitwise encoding of the first int of the first example:')
                    mlb.log(f'ints_embedded :: {tuple(ints_embedded.shape)} :: (ints, longest_num_digits, H)')
                    mlb.log(f'Heres the ints_embedded[0] which is the digitwise encoding of the first int of the first example. You may see the zero padding. :')
                    mlb.log(ints_embedded[0])
                    packed, unsorter = self.digit_embedder.sort_and_pack(ints_embedded, digits_per_int)
                    mlb.log('running thru int_encoder() gru')
                    _, hidden = self.int_encoder(packed)
                    mlb.log(f'unsorting by unsorter: {unsorter.tolist()}')
                    int_encodings = hidden.sum(0)[unsorter] # sum over bidirectionality [num_exs*list_len, H]
                    mlb.log(f'int_encodings :: {tuple(int_encodings.shape)} :: (ints, H)')
                    mlb.log(f'int_encodings[:3] the first 3 int encodings (feed in dummies to make sure they come out right!):')
                    mlb.log(int_encodings[:3])
                else:
                    int_encodings = None
                # undo the flattening
                pad_till = max(ints_per_ex)+2 # +2 for LIST_START and LIST_END
                # make our examplewise tensor with padding already inserted as zeros
                exs_tensor = torch.zeros(len(exs),pad_till,self.H) # [num_exs,longest_list_len,H]
                exs_tensor = exs_tensor.to(self.device)
                j = 0
                # add LIST_START
                list_start_vec = self.digit_embedder(self.digit_embedder.idx_of_tok['LIST_START'])
                list_end_vec = self.digit_embedder(self.digit_embedder.idx_of_tok['LIST_END'])
                exs_tensor[:,0] = list_start_vec.expand(len(exs),-1)
                for i,num_ints in enumerate(ints_per_ex):
                    if int_encodings is not None: # only None if the input was an empty list of ints
                        exs_tensor[i,1:num_ints+1] = int_encodings[j:j+num_ints]
                    exs_tensor[i,num_ints+1] = list_end_vec # add LIST_END
                    j += num_ints
                sizes = torch.tensor([x+2 for x in ints_per_ex]) # +2 bc list start and list end
                # exs_tensor :: [num_exs, longest_list_len, H]
                mlb.log(f'exs_tensor :: {tuple(exs_tensor.shape)} :: (num_exs, longest_list_len, H)')
                mlb.log(f'note that at this point LIST_START and LIST_END were added, along with lots of zero padding')
                mlb.log(f'the first example is exs_tensor[0]:')
                mlb.log(exs_tensor[0])
                assert exs_tensor[0][0].equal(exs_tensor[1][0]) # both shd be LIST_START
                packed,unsorter = self.digit_embedder.sort_and_pack(exs_tensor,sizes)
                mlb.log(f'running list_encoder gru')
                _, hidden = self.list_encoder(packed)
                mlb.log(f'unsorting with unsorter: {unsorter.tolist()}')
                list_encodings = hidden.sum(0)[unsorter] # sum over bidirectionaliy. [num_exs,H]
                mlb.log(f'encodeValue() is returning list_encodings :: {tuple(list_encodings.shape)} :: (num_exs,H)')
                return list_encodings

            # [non-digitwise]
            indices_lists = []
            for int_list in val:
                tokens = ["LIST_START"] + int_list + ["LIST_END"]
                assert len(tokens) <= self.maximumLength
                indices = self.tokensToIndices(tokens) # [int]
                indices_lists.append(indices)
            
            # pad
            indices_lists, sizes = self.pad_indices_lists(indices_lists) # [[int]]
            # sort by sizes for pytorch efficiency
            sizes,sorter = sizes.sort(descending=True)
            _,unsorter = sorter.sort() # fun trick
            # embed & pack
            embeddings = self.embedding(indices_lists, cuda=self.use_cuda) # [num_exs, padded_ex_length, H]
            embeddings = embeddings[sorter] # sort by decreasing size
            embeddings = embeddings.permute(1,0,2) # swap first two dims. [padded_ex_length, num_exs, H]
            packed = pack_padded_sequence(embeddings,sizes)

            _, hidden = self.list_encoder(packed)
            res = hidden.sum(0) # sum over bidirectionality. [num_exs, H]
            sizes = sizes[unsorter]
            res = res[unsorter] # undo the sorting so the examples line up with what were passed in originally
            return res
            # [END non-digitwise]

        # [non-list]
        if is_bool(val[0]):
            indices = torch.tensor(self.digit_embedder.idxs_of_toks(val)) # [int]
            indices = indices.to(self.device)
            res = self.digit_embedder(indices)
            return res
        

        assert is_int(val[0])
        # val is one concrete non-list value per example
        # no LIST_START or anything needed here! Bc the list is examplewise

        if self.digitwise:
            if is_int(val[0]):
                ints = val # :: [num_exs,]
                ints = [['INT_START']+list(str(_int))+['INT_END'] for _int in ints] # convert 100 -> ['1','0','0']
                before_idxs_of_toks = ints # in case you want to inspect during debugging
                ints = self.digit_embedder.idxs_of_toks(ints)
                # pad it to a constant number of digits per int
                ints, digits_per_int = self.digit_embedder.pad_idxs_lists(ints)
                ints = torch.stack([torch.stack(digits) for digits in ints]) # :: [num_exs, longest_num_digits]
                ints_embedded = self.digit_embedder(ints) # pointwise, converts indices -> embeddings
                packed, unsorter = self.digit_embedder.sort_and_pack(ints_embedded, digits_per_int)
                _, hidden = self.int_encoder(packed)
                int_encodings = hidden.sum(0)[unsorter] # sum over bidirectionality [num_exs*list_len, H]
                return int_encodings
            # it's a list of functions or other non-int things
            idxs = torch.stack(self.digit_embedder.idxs_of_toks(val))
            res = self.digit_embedder(idxs)
            return res

        # [non-digitwise]
        indices = self.tokensToIndices(val) # [int]
        res = self.embedding(indices, cuda=self.use_cuda) # [num_exs,H]
        return res

    # def sampleHelmholtzTask(self, request, motifs=[]):
    #     assert False # disabling for now
    #     # NOTE: we ignore the `request` argument
    #     if motifs != []:
    #         raise NotImplementedError
    #     try:
    #         program, task = self.deepcoder_taskloader.getTask()
    #     except StopIteration:
    #         return None,None
    #     return program, task



def train_necessary(t):
    if t.name in {"head", "is-primes", "len", "pop", "repeat-many", "tail", "keep primes", "keep squares"}:
        return True
    if any(t.name.startswith(x) for x in {
        "add-k", "append-k", "bool-identify-geq-k", "count-k", "drop-k",
        "empty", "evens", "has-k", "index-k", "is-mod-k", "kth-largest",
        "kth-smallest", "modulo-k", "mult-k", "remove-index-k",
        "remove-mod-k", "repeat-k", "replace-all-with-index-k", "rotate-k",
        "slice-k-n", "take-k",
    }):
        return "some"
    return False


def list_options(parser):
    parser.add_argument(
        "--noMap", action="store_true", default=False,
        help="Disable built-in map primitive")
    parser.add_argument(
        "--noUnfold", action="store_true", default=False,
        help="Disable built-in unfold primitive")
    parser.add_argument(
        "--noLength", action="store_true", default=False,
        help="Disable built-in length primitive")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Lucas-old",
        choices=[
            "bootstrap",
            "sorting",
            "Lucas-old",
            "Lucas-depth1",
            "Lucas-depth2",
            "Lucas-depth3"])
    parser.add_argument("--maxTasks", type=int,
                        default=None,
                        help="truncate tasks to fit within this boundary")
    parser.add_argument("--primitives",
                        default="common",
                        help="Which primitive set to use",
                        choices=["McCarthy", "base", "rich", "common", "noLength", "deepcoder"])
    parser.add_argument("--extractor", type=str,
                        choices=["hand", "deep", "learned"],
                        default="learned")
    parser.add_argument("--split", metavar="TRAIN_RATIO",
                        type=float,
                        help="split test/train")
    parser.add_argument("-H", "--hidden", type=int,
                        default=64,
                        help="number of hidden units")
    parser.add_argument("--random-seed", type=int, default=17)


def main(args):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on manipulating sequences of numbers.
    """
    random.seed(args.pop("random_seed"))

    dataset = args.pop("dataset")
    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks("data/list_tasks.json") + sortBootstrap(),
        "bootstrap": make_list_bootstrap_tasks,
        "sorting": sortBootstrap,
        "Lucas-depth1": lambda: retrieveJSONTasks("data/list_tasks2.json")[:105],
        "Lucas-depth2": lambda: retrieveJSONTasks("data/list_tasks2.json")[:4928],
        "Lucas-depth3": lambda: retrieveJSONTasks("data/list_tasks2.json"),
    }[dataset]()

    maxTasks = args.pop("maxTasks")
    if maxTasks and len(tasks) > maxTasks:
        necessaryTasks = []  # maxTasks will not consider these
        if dataset.startswith("Lucas2.0") and dataset != "Lucas2.0-depth1":
            necessaryTasks = tasks[:105]

        eprint("Unwilling to handle {} tasks, truncating..".format(len(tasks)))
        random.shuffle(tasks)
        del tasks[maxTasks:]
        tasks = necessaryTasks + tasks

    if dataset.startswith("Lucas"):
        # extra tasks for filter
        tasks.extend([
            Task("remove empty lists",
                 arrow(tlist(tlist(tbool)), tlist(tlist(tbool))),
                 [((ls,), list(filter(lambda l: len(l) > 0, ls)))
                  for _ in range(15)
                  for ls in [[[random.random() < 0.5 for _ in range(random.randint(0, 3))]
                              for _ in range(4)]]]),
            Task("keep squares",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: int(math.sqrt(x)) ** 2 == x,
                                      xs)))
                  for _ in range(15)
                  for xs in [[random.choice([0, 1, 4, 9, 16, 25])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
            Task("keep primes",
                 arrow(tlist(tint), tlist(tint)),
                 [((xs,), list(filter(lambda x: x in {2, 3, 5, 7, 11, 13, 17,
                                                      19, 23, 29, 31, 37}, xs)))
                  for _ in range(15)
                  for xs in [[random.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
                              if random.random() < 0.5
                              else random.randint(0, 9)
                              for _ in range(7)]]]),
        ])
        for i in range(4):
            tasks.extend([
                Task("keep eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x == i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove eq %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x != i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("keep gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]]),
                Task("remove gt %s" % i,
                     arrow(tlist(tint), tlist(tint)),
                     [((xs,), list(filter(lambda x: not x > i, xs)))
                      for _ in range(15)
                      for xs in [[random.randint(0, 6) for _ in range(5)]]])
            ])

    def isIdentityTask(t):
        return all( len(xs) == 1 and xs[0] == y for xs, y in t.examples  )
    eprint("Removed", sum(isIdentityTask(t) for t in tasks), "tasks that were just the identity function")
    tasks = [t for t in tasks if not isIdentityTask(t) ]

    prims = {"base": basePrimitives,
             "McCarthy": McCarthyPrimitives,
             "common": bootstrapTarget_extra,
             "noLength": no_length,
             "deepcoder": deepcoderPrimitives,
             "rich": primitives}[args.pop("primitives")]()
    haveLength = not args.pop("noLength")
    haveMap = not args.pop("noMap")
    haveUnfold = not args.pop("noUnfold")
    eprint(f"Including map as a primitive? {haveMap}")
    eprint(f"Including length as a primitive? {haveLength}")
    eprint(f"Including unfold as a primitive? {haveUnfold}")
    baseGrammar = Grammar.uniform([p
                                   for p in prims
                                   if (p.name != "map" or haveMap) and \
                                   (p.name != "unfold" or haveUnfold) and \
                                   (p.name != "length" or haveLength)])

    extractor = {
        "learned": LearnedFeatureExtractor,
    }[args.pop("extractor")]
    extractor.H = args.pop("hidden")

    timestamp = datetime.datetime.now().isoformat()
    outputDirectory = "experimentOutputs/list/%s"%timestamp
    os.system("mkdir -p %s"%outputDirectory)
    
    args.update({
        "featureExtractor": extractor,
        "outputPrefix": "%s/list"%outputDirectory,
        "evaluationTimeout": 0.0005,
    })
    

    eprint("Got {} list tasks".format(len(tasks)))
    split = args.pop("split")
    if split:
        train_some = defaultdict(list)
        for t in tasks:
            necessary = train_necessary(t)
            if not necessary:
                continue
            if necessary == "some":
                train_some[t.name.split()[0]].append(t)
            else:
                t.mustTrain = True
        for k in sorted(train_some):
            ts = train_some[k]
            random.shuffle(ts)
            ts.pop().mustTrain = True

        test, train = testTrainSplit(tasks, split)
        if True:
            test = [t for t in test
                    if t.name not in EASYLISTTASKS]

        eprint(
            "Alotted {} tasks for training and {} for testing".format(
                len(train), len(test)))
    else:
        train = tasks
        test = []
    
    T=3
    testing_taskloader = DeepcoderTaskloader(
        #'dreamcoder/domains/list/DeepCoder_data/T1_A2_V512_L10_test_perm.txt', #TODO <- careful this is set to `train` instead of `test` for an ultra simple baseline
        #'dreamcoder/domains/list/DeepCoder_data/T1_A2_V512_L10_test_perm.txt',
        #'dreamcoder/domains/list/DeepCoder_data/T2_A2_V512_L10_test_perm.txt',
        f'dreamcoder/domains/list/DeepCoder_data/T{T}_A2_V512_L10_test_perm.txt',
        allowed_requests=[arrow(tlist(tint),tlist(tint))],
        repeat=True,
        micro=None,
        )

    test = [testing_taskloader.getTask()[1] for _ in range(8)]

    explorationCompression(baseGrammar,[], testingTasks=test, **args)
