from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import copy

#A syntax-checking Robustfill, inspired by https://arxiv.org/pdf/1805.04276.pdf (Leveraging grammar and reinforcement learning ...)

def choose(matrix, idxs):
    if type(idxs) is Variable: idxs = idxs.data
    assert(matrix.ndimension()==2)
    unrolled_idxs = idxs + torch.arange(0, matrix.size(0)).type_as(idxs)*matrix.size(1)
    return matrix.view(matrix.nelement())[unrolled_idxs]

class SyntaxCheckingRobustFill(nn.Module):
    def __init__(self, input_vocabularies, target_vocabulary, hidden_size=512, embedding_size=128, cell_type="LSTM", max_length=25):
        """
        Terminology is a little confusing. The SyntaxCheckingRobustFill is the full model, which contains a SyntaxLSTM inside it
        :param: input_vocabularies: List containing a vocabulary list for each input. E.g. if learning a function f:A->B from (a,b) pairs, input_vocabularies has length 2
        :param: target_vocabulary: Vocabulary list for output
        """
        super(SyntaxCheckingRobustFill, self).__init__()
        self.n_encoders = len(input_vocabularies)

        self.t = Parameter(torch.ones(1)) #template

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_vocabularies = input_vocabularies
        self.target_vocabulary = target_vocabulary
        self._refreshVocabularyIndex()
        self.v_inputs = [len(x) for x in input_vocabularies] # Number of tokens in input vocabularies
        self.v_target = len(target_vocabulary) # Number of tokens in target vocabulary

        self.no_inputs = len(self.input_vocabularies)==0
        self.max_length = max_length

        self.cell_type=cell_type
        if cell_type=='GRU':
            self.encoder_init_h = Parameter(torch.rand(1, self.hidden_size))
            self.encoder_cells = nn.ModuleList(
                [nn.GRUCell(input_size=self.v_inputs[0]+1, hidden_size=self.hidden_size, bias=True)] + 
                [nn.GRUCell(input_size=self.v_inputs[i]+1+self.hidden_size, hidden_size=self.hidden_size, bias=True) for i in range(1, self.n_encoders)]
            )
            self.decoder_cell = nn.GRUCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)

            self.syntax_decoder_cell = nn.GRUCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)

        if cell_type=='LSTM':
            self.encoder_init_h = Parameter(torch.rand(1, self.hidden_size)) #Also used for decoder if self.no_inputs=True
            self.encoder_init_cs = nn.ParameterList(
                [Parameter(torch.rand(1, self.hidden_size)) for i in range(len(self.v_inputs))]
            )
            self.encoder_cells = nn.ModuleList()
            for i in range(self.n_encoders):
                input_size = self.v_inputs[i] + 1 + (self.hidden_size if i>0 else 0)
                self.encoder_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=self.hidden_size, bias=True))
            self.decoder_cell = nn.LSTMCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)
            self.decoder_init_c = Parameter(torch.rand(1, self.hidden_size))

            self.syntax_decoder_cell = nn.LSTMCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)

            self.syntax_decoder_init_c = Parameter(torch.rand(1, self.hidden_size))
            self.syntax_decoder_init_h = Parameter(torch.rand(1, self.hidden_size))
        
        self.W = nn.Linear(self.hidden_size if self.no_inputs else 2*self.hidden_size, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_target+1)

        self.syntax_W = nn.Linear(self.hidden_size, self.embedding_size)
        self.syntax_V = nn.Linear(self.embedding_size, self.v_target+1)

        self.As = nn.ModuleList([nn.Bilinear(self.hidden_size, self.hidden_size, 1, bias=False) for i in range(self.n_encoders)])

        #SYNTAX CHECKER LSTM:
        #self.SyntaxLSTM = RobustFill([], target_vocabulary, hidden_size, embedding_size, cell_type, max_length)

        #rewrite SyntaxLSTM run function so that I have access to the whole thing




    def with_target_vocabulary(self, target_vocabulary):
        """
        Returns a new network which modifies this one by changing the target vocabulary
        """
        print("WARNING: with_target_vocabulary not yet tested for syntax checker RobsutFill")
        #assert False 

        if target_vocabulary == self.target_vocabulary:
            return self

        V_weight = []
        V_bias = []
        decoder_ih = []

        #syntax
        syntax_V_weight = []
        syntax_V_bias = []
        syntax_decoder_ih = []

        for i in range(len(target_vocabulary)):
            if target_vocabulary[i] in self.target_vocabulary:
                j = self.target_vocabulary.index(target_vocabulary[i])
                V_weight.append(self.V.weight.data[j:j+1])
                V_bias.append(self.V.bias.data[j:j+1])
                decoder_ih.append(self.decoder_cell.weight_ih.data[:,j:j+1])

                #syntax
                syntax_V_weight.append(self.syntax_V.weight.data[j:j+1])
                syntax_V_bias.append(self.syntax_V.bias.data[j:j+1])
                syntax_decoder_ih.append(self.syntax_decoder_cell.weight_ih.data[:,j:j+1])

            else:
                V_weight.append(self._zeros(1, self.V.weight.size(1)))
                V_bias.append(self._ones(1) * -10)
                decoder_ih.append(self._zeros(self.decoder_cell.weight_ih.data.size(0), 1))

                #syntax
                syntax_V_weight.append(self._zeros(1, self.syntax_V.weight.size(1)))
                syntax_V_bias.append(self._ones(1) * -10)
                syntax_decoder_ih.append(self._zeros(self.syntax_decoder_cell.weight_ih.data.size(0), 1))



        V_weight.append(self.V.weight.data[-1:])
        V_bias.append(self.V.bias.data[-1:])
        decoder_ih.append(self.decoder_cell.weight_ih.data[:,-1:])

        #syntax
        syntax_V_weight.append(self.syntax_V.weight.data[-1:])
        syntax_V_bias.append(self.syntax_V.bias.data[-1:])
        syntax_decoder_ih.append(self.syntax_decoder_cell.weight_ih.data[:,-1:])



        self.target_vocabulary = target_vocabulary
        self.v_target = len(target_vocabulary)

        self.V.weight.data = torch.cat(V_weight, dim=0)
        self.V.bias.data = torch.cat(V_bias, dim=0)
        self.V.out_features = self.V.bias.data.size(0)

        self.decoder_cell.weight_ih.data = torch.cat(decoder_ih, dim=1)
        self.decoder_cell.input_size = self.decoder_cell.weight_ih.data.size(1)

        #syntax
        self.syntax_V.weight.data = torch.cat(syntax_V_weight, dim=0)
        self.syntax_V.bias.data = torch.cat(syntax_V_bias, dim=0)
        self.syntax_V.out_features = self.syntax_V.bias.data.size(0)

        self.syntax_decoder_cell.weight_ih.data = torch.cat(syntax_decoder_ih, dim=1)
        self.syntax_decoder_cell.input_size = self.syntax_decoder_cell.weight_ih.data.size(1)


        self._clear_optimiser()
        self._refreshVocabularyIndex()
        return copy.deepcopy(self)

        #decoder_cell
        #self.syntax_V
        #self.syntax_decoder_cell

    def optimiser_step(self, batch_inputs, batch_target, vocab_filter=None):
        """
        Perform a single step of SGD
        """
        if not hasattr(self, 'opt'): self._get_optimiser()
        self.opt.zero_grad()
        score, syntax_score = self.score(batch_inputs, batch_target, autograd=True, vocab_filter=vocab_filter)
        score = score.mean()
        syntax_score = syntax_score.mean()

        (-score - syntax_score).backward()
        self.opt.step()
                
        return score.data.item(), syntax_score.data.item()

    def score(self, batch_inputs, batch_target, autograd=False, vocab_filter=None):
        inputs = self._inputsToTensors(batch_inputs)
        target = self._targetToTensor(batch_target)
        _, score, syntax_score = self._run(inputs, target=target, mode="score", vocab_filter=vocab_filter)
        if autograd:
            return score, syntax_score
        else:
            return score.data, syntax_score.data

    def sample(self, batch_inputs=None, n_samples=None, vocab_filter=None):
        assert batch_inputs is not None or n_samples is not None
        inputs = self._inputsToTensors(batch_inputs)
        target, score, syntax_score = self._run(inputs, mode="sample", n_samples=n_samples, vocab_filter=vocab_filter)
        target = self._tensorToOutput(target)
        return target

    def sampleAndScore(self, batch_inputs=None, n_samples=None, nRepeats=None, autograd=False, vocab_filter=None):
        assert batch_inputs is not None or n_samples is not None
        inputs = self._inputsToTensors(batch_inputs)
        if nRepeats is None:
            target, score, syntax_score = self._run(inputs, mode="sample", n_samples=n_samples, vocab_filter=vocab_filter)
            target = self._tensorToOutput(target)
            return target, score.data, syntax_score.data
        else:
            target = []
            score = []
            syntax_score = []
            for i in range(nRepeats):
                t, s, ss = self._run(inputs, mode="sample", n_samples=n_samples, vocab_filter=vocab_filter)
                t = self._tensorToOutput(t)
                target.extend(t)
                if autograd:
                    score.extend(list(s))
                    syntax_score.extend(list(ss))
                else:
                    score.extend(list(s.data))
                    syntax_score.extend(list(ss.data))
            return target, score, syntax_score
                                
    def _refreshVocabularyIndex(self): #TODO
        self.input_vocabularies_index = [
            {self.input_vocabularies[i][j]: j for j in range(len(self.input_vocabularies[i]))}
            for i in range(len(self.input_vocabularies))
        ]
        self.target_vocabulary_index = {self.target_vocabulary[j]: j for j in range(len(self.target_vocabulary))}
        
    def __getstate__(self):
        if hasattr(self, 'opt'):
            return dict([(k,v) for k,v in self.__dict__.items() if k is not 'opt'] + 
                        [('optstate', self.opt.state_dict())])
        else: return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'optstate'): self._fix_optstate()

    def _ones(self, *args, **kwargs):
        return self.t.new_ones(*args, **kwargs)

    def _zeros(self, *args, **kwargs):
        return self.t.new_zeros(*args, **kwargs)

    def _clear_optimiser(self):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): del self.optstate

    def _get_optimiser(self, lr=0.001):
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        if hasattr(self, 'optstate'): self.opt.load_state_dict(self.optstate)

    def _fix_optstate(self): #make sure that we don't have optstate on as tensor but params as cuda tensor, or vice versa
        is_cuda = next(self.parameters()).is_cuda
        for state in self.optstate['state'].values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda() if is_cuda else v.cpu()

    def cuda(self, *args, **kwargs):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): self._fix_optstate()
        super(SyntaxCheckingRobustFill, self).cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): self._fix_optstate()
        super(SyntaxCheckingRobustFill, self).cpu(*args, **kwargs)

    def _encoder_get_init(self, encoder_idx, h=None, batch_size=None):
        if h is None: h = self.encoder_init_h.repeat(batch_size, 1)
        if self.cell_type=="GRU": return h
        if self.cell_type=="LSTM": return (h, self.encoder_init_cs[encoder_idx].repeat(batch_size, 1))

    def _decoder_get_init(self, h=None, batch_size=None):
        if h is None:
            assert self.no_inputs
            h = self.encoder_init_h.repeat(batch_size, 1)

        if self.cell_type=="GRU": return h
        if self.cell_type=="LSTM": return (h, self.decoder_init_c.repeat(h.size(0), 1))

    def _syntax_decoder_get_init(self, h=None, batch_size=None):
        assert h==None
        h = self.syntax_decoder_init_h.repeat(batch_size, 1) #TODO

        if self.cell_type=="GRU": return h
        if self.cell_type=="LSTM": return (h, self.syntax_decoder_init_c.repeat(h.size(0), 1)) #TODO

    def _cell_get_h(self, cell_state):
        if self.cell_type=="GRU": return cell_state
        if self.cell_type=="LSTM": return cell_state[0]

    def _run(self, inputs, target=None, mode="sample", n_samples=None, vocab_filter=None):
        """
        :param mode: "score" or "sample"
        :param list[list[LongTensor]] inputs: n_encoders * n_examples * (max length * batch_size)
        :param list[LongTensor] target: max length * batch_size
        :param vocab_filter: batch_size * ... (set of possible outputs)
        Returns output and score
        """
        assert((mode=="score" and target is not None) or mode=="sample" or mode=="encode_only")



        if vocab_filter is not None:
            vocab_mask = self.t.new([[v in V for v in self.target_vocabulary] + [True] for V in vocab_filter]).byte() #True for STOP

        if self.no_inputs:
            batch_size = target.size(1) if mode=="score" else n_samples
        else:
            batch_size = inputs[0][0].size(1)
            n_examples = len(inputs[0])
            max_length_inputs = [[inputs[i][j].size(0) for j in range(n_examples)] for i in range(self.n_encoders)]
            inputs_scatter = [
                [   Variable(self._zeros(max_length_inputs[i][j], batch_size, self.v_inputs[i]+1).scatter_(2, inputs[i][j][:, :, None], 1))
                    for j in range(n_examples)
                ] for i in range(self.n_encoders)
            ]  # n_encoders * n_examples * (max_length_input * batch_size * v_input+1)

        if mode=="encode_only": assert batch_size == 1 #for now

        max_length_target = target.size(0) if target is not None else self.max_length
        score = Variable(self._zeros(batch_size))
        syntax_score = Variable(self._zeros(batch_size))
        if target is not None: target_scatter = Variable(self._zeros(max_length_target, batch_size, self.v_target+1).scatter_(2, target[:, :, None], 1)) # max_length_target * batch_size * v_target+1

        H = [] # n_encoders * n_examples * (max_length_input * batch_size * h_encoder_size)
        embeddings = [] # n_encoders * (h for example at INPUT_EOS)
        attention_mask = [] # n_encoders * (0 until (and including) INPUT_EOS, then -inf)
        def attend(i, j, h):
            """
            'general' attention from https://arxiv.org/pdf/1508.04025.pdf
            :param i: which encoder is doing the attending (or self.n_encoders for the decoder)
            :param j: Index of example
            :param h: batch_size * hidden_size
            """
            assert(i != 0)
            scores = self.As[i-1](
                H[i-1][j].view(max_length_inputs[i-1][j] * batch_size, self.hidden_size),
                h.view(batch_size, self.hidden_size).repeat(max_length_inputs[i-1][j], 1)
            ).view(max_length_inputs[i-1][j], batch_size) + attention_mask[i-1][j]
            c = (F.softmax(scores[:, :, None], dim=0) * H[i-1][j]).sum(0)
            return c


        # -------------- Encoders -------------
        for i in range(len(self.input_vocabularies)):
            H.append([])
            embeddings.append([])
            attention_mask.append([])

            for j in range(n_examples):
                active = self._ones(max_length_inputs[i][j], batch_size).byte()
                state = self._encoder_get_init(i, batch_size=batch_size, h=embeddings[i-1][j] if i>0 else None)
                hs = []
                h = self._cell_get_h(state)
                for k in range(max_length_inputs[i][j]):
                    if i==0:
                        state = self.encoder_cells[i](inputs_scatter[i][j][k, :, :], state)
                    else:
                        state = self.encoder_cells[i](torch.cat([inputs_scatter[i][j][k, :, :], attend(i, j, h)], 1), state)
                    if k+1 < max_length_inputs[i][j]: active[k+1, :] = active[k, :] * (inputs[i][j][k, :] != self.v_inputs[i])
                    h = self._cell_get_h(state) 
                    hs.append(h[None, :, :])
                H[i].append(torch.cat(hs, 0))
                embedding_idx = active.sum(0).long() - 1
                embedding = H[i][j].gather(0, Variable(embedding_idx[None, :, None].repeat(1, 1, self.hidden_size)))[0]
                embeddings[i].append(embedding)
                attention_mask[i].append(Variable(active.float().log()))


        # ------------------ Decoder -----------------
        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        target = target if mode=="score" else self._zeros(max_length_target, batch_size).long()
        if self.no_inputs: decoder_states = [self._decoder_get_init(batch_size=batch_size)] #TODO: learn seperate one of these 
        else: decoder_states = [self._decoder_get_init(embeddings[self.n_encoders-1][j]) for j in range(n_examples)] #P
        syntax_decoder_state = self._syntax_decoder_get_init(batch_size=batch_size) #TODO
        active = self._ones(batch_size).byte()

        #holy hell what a hack
        if mode=="encode_only": return target, score, decoder_states, syntax_decoder_state, active, H, attention_mask, max_length_inputs, batch_size, n_examples

        for k in range(max_length_target):
            FC = []
            #syntax_FC = []
            for j in range(1 if self.no_inputs else n_examples):
                h = self._cell_get_h(decoder_states[j])
                p_aug = h if self.no_inputs else torch.cat([h, attend(self.n_encoders, j, h)], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))

            #Syntax:
            syntax_p_aug = self._cell_get_h(syntax_decoder_state)
            syntax_m = F.tanh(self.syntax_W(syntax_p_aug))
            #syntax_FC.append(F.tanh(self.syntax_W(syntax_p_aug)[None, :, :])) #TODO


            #Here
            #print("FC size", FC[0].size())
            m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
            #print("m size", m.size())

            v = self.V(m)

            #print("syntax_m size", syntax_m.size())
            syntax_v = self.syntax_V(syntax_m) #TODO
            #Syntax checker term:
            syntax_logsoftmax = F.log_softmax(syntax_v, dim=1)
            #bug: the below line only works in score mode, and not sample mode, because target hasn't been defined in sample mode yet
            syntax_score = syntax_score + choose(syntax_logsoftmax, target[k, :]) * Variable(active.float())

            v = v + syntax_logsoftmax
            if vocab_filter is not None: v = v.masked_fill(1-vocab_mask, float('-inf'))
            logsoftmax = F.log_softmax(v, dim=1)
            if mode=="sample": target[k, :] = torch.multinomial(logsoftmax.data.exp(), 1)[:, 0]


            #this is where beam stuff goes ... 
            score = score + choose(logsoftmax, target[k, :]) * Variable(active.float())



            active *= (target[k, :] != self.v_target)
            for j in range(1 if self.no_inputs else n_examples):
                if mode=="score":
                    target_char_scatter = target_scatter[k, :, :]
                elif mode=="sample":
                    target_char_scatter = Variable(self._zeros(batch_size, self.v_target+1).scatter_(1, target[k, :, None], 1))
                decoder_states[j] = self.decoder_cell(target_char_scatter, decoder_states[j]) 
            syntax_decoder_state = self.syntax_decoder_cell(target_char_scatter, syntax_decoder_state) #TODO
        return target, score, syntax_score

        """
        score
        active
        target_scatter
        target_char_scatter
        decoder_states[j]
        syntax_decoder_state
        
        """


    def _inputsToTensors(self, inputsss):
        """
        :param inputs: size = nBatch * nExamples * nEncoders (or nBatch*nExamples is n_encoders=1)
        Returns nEncoders * nExamples tensors of size nBatch * max_len
        """
        if self.n_encoders == 0: return []
        tensors = []
        for i in range(self.n_encoders):
            tensors.append([])
            for j in range(len(inputsss[0])):
                if self.n_encoders == 1:
                    inputs = [x[j] for x in inputsss]
                else: inputs = [x[j][i] for x in inputsss]
                maxlen = max(len(s) for s in inputs)
                t = self._ones(maxlen+1, len(inputs)).long()*self.v_inputs[i]
                for k in range(len(inputs)):
                    s = inputs[k]
                    try:
                        if len(s)>0: t[:len(s), k] = torch.LongTensor([self.input_vocabularies_index[i][x] for x in s])
                    except KeyError: import pdb; pdb.set_trace()
                tensors[i].append(t)
        return tensors

    def _targetToTensor(self, targets):
        """
        :param targets: 
        """
        maxlen = max(len(s) for s in targets)
        t = self._ones(maxlen+1, len(targets)).long()*self.v_target
        for i in range(len(targets)):
            s = targets[i]
            if len(s)>0: t[:len(s), i] = torch.LongTensor([self.target_vocabulary_index[x] for x in s])
        return t

    def _tensorToOutput(self, tensor):
        """
        :param tensor:
        """
        out = []
        for i in range(tensor.size(1)):
            l = tensor[:,i].tolist()
            if l[0]==self.v_target:
                out.append(tuple())
            elif self.v_target in l:
                final = tensor[:,i].tolist().index(self.v_target)
                out.append(tuple(self.target_vocabulary[x] for x in tensor[:final, i]))
            else:
                out.append(tuple(self.target_vocabulary[x] for x in tensor[:, i]))
        return out  



    def beam_decode(self, batch_inputs=None, beam_size=None, vocab_filter=None, maxlen=None):
        
        inputs = self._inputsToTensors(batch_inputs)

        beam = self._run_with_beam(inputs, beam_size=beam_size, vocab_filter=vocab_filter, maxlen=maxlen)
        outputs = list(zip(*beam))
        target_tensors, scores = outputs[0], outputs[1] #oy

        targets = []
        for target in target_tensors:
            targets.extend(self._tensorToOutput(target))
        return targets, [score.data for score in scores] #might want a .data here



    def _run_with_beam(self, inputs, beam_size=10, vocab_filter=None, maxlen=None):
        triggered = False
        #assert batchsize is 1 for now

        #encode to decoder state
        target, score, decoder_states, syntax_decoder_state, active, H, attention_mask, max_length_inputs, batch_size, n_examples = self._encode(inputs, vocab_filter=vocab_filter) #use hack on run
        beam = [(target, score, decoder_states, syntax_decoder_state, active)] 

        max_len = maxlen if maxlen is not None else self.max_length
        for k in range(max_len):
            new_beam = []
            for target, score, decoder_states, syntax_decoder_state, active in beam:
                if not any(active==True):
                    if len(new_beam) < beam_size:
                        new_beam.append((target, score, decoder_states, syntax_decoder_state, active))  # I think it's fine not to clone here
                        new_beam = sorted(new_beam, key=lambda entry: -entry[1])
                    else: #len >= beam_size
                        if score > new_beam[-1][1]: #the worst score in new_beam
                            #replace
                            new_beam[-1] = (target, score, decoder_states, syntax_decoder_state, active)
                            new_beam = sorted(new_beam, key=lambda entry: -entry[1])
                else:
                    #print(k, flush=True)
                    logsoftmax = self._run_first_half(k, decoder_states, syntax_decoder_state, H, attention_mask, max_length_inputs, batch_size, n_examples, vocab_filter=vocab_filter)
                    for token in list(self.target_vocabulary_index.values()) + [self.v_target]:
                        #do filtering for these lines 
                        candidate = self._run_second_half(k, logsoftmax, target.clone(), token, score.clone(), [(ds[0].clone(), ds[1].clone()) for ds in decoder_states], (syntax_decoder_state[0].clone(), syntax_decoder_state[1].clone()), active.clone(), batch_size, n_examples)
                        if len(new_beam) < beam_size:
                            new_beam.append(candidate)
                            new_beam = sorted(new_beam, key=lambda entry: -entry[1])
                        else: #len >= beam_size
                            if candidate[1] > new_beam[-1][1]: #the worst score in new_beam
                                new_beam[-1] = candidate
                                new_beam = sorted(new_beam, key=lambda entry: -entry[1])
                        
            #new_beam = sorted(new_beam, key=lambda entry: -entry[1]) # i think this is right....
            beam = new_beam
            assert len(beam) <= beam_size
        return beam


    def _encode(self, inputs, vocab_filter=None):
        return self._run(inputs, target=None, mode="encode_only", n_samples=None, vocab_filter=vocab_filter)


    def attend_for_beam(self, i, j, h, H, attention_mask, max_length_inputs, batch_size):
        """
        'general' attention from https://arxiv.org/pdf/1508.04025.pdf
        :param i: which encoder is doing the attending (or self.n_encoders for the decoder)
        :param j: Index of example
        :param h: batch_size * hidden_size
        """
        assert(i != 0)
        scores = self.As[i-1](
            H[i-1][j].view(max_length_inputs[i-1][j] * batch_size, self.hidden_size),
            h.view(batch_size, self.hidden_size).repeat(max_length_inputs[i-1][j], 1)
        ).view(max_length_inputs[i-1][j], batch_size) + attention_mask[i-1][j]
        c = (F.softmax(scores[:, :, None], dim=0) * H[i-1][j]).sum(0)
        return c

    def _run_first_half(self, k, decoder_states, syntax_decoder_state, H, attention_mask, max_length_inputs, batch_size, n_examples, vocab_filter=None):

        FC = []
        #syntax_FC = []
        for j in range(1 if self.no_inputs else n_examples):
            h = self._cell_get_h(decoder_states[j])
            p_aug = h if self.no_inputs else torch.cat([h, self.attend_for_beam(self.n_encoders, j, h, H, attention_mask, max_length_inputs, batch_size)], 1)  #will need H and attention_mask for attend fn
            FC.append(F.tanh(self.W(p_aug)[None, :, :]))

        #Syntax:
        syntax_p_aug = self._cell_get_h(syntax_decoder_state)
        syntax_m = F.tanh(self.syntax_W(syntax_p_aug))
        #syntax_FC.append(F.tanh(self.syntax_W(syntax_p_aug)[None, :, :])) #TODO


        #Here
        #print("FC size", FC[0].size())
        m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
        #print("m size", m.size())

        v = self.V(m)

        #print("syntax_m size", syntax_m.size())
        syntax_v = self.syntax_V(syntax_m) #TODO
        #Syntax checker term:
        syntax_logsoftmax = F.log_softmax(syntax_v, dim=1)

        #bug: the below line only works in score mode, and not sample mode
        #syntax_score = syntax_score + choose(syntax_logsoftmax, target[k, :]) * Variable(active.float())


        v = v + syntax_logsoftmax
        if vocab_filter is not None: v = v.masked_fill(1-vocab_mask, float('-inf'))
        logsoftmax = F.log_softmax(v, dim=1)
        #if mode=="sample": target[k, :] = torch.multinomial(logsoftmax.data.exp(), 1)[:, 0]

        return logsoftmax.clone()


    def _run_second_half(self, k, logsoftmax, target, token, score, decoder_states, syntax_decoder_state, active, batch_size, n_examples):


        #reference: t[:,i] = torch.Tensor([token]), where i is batch index
        target[k, :] = torch.Tensor([token]*batch_size) #just put it on there

        #this is where beam stuff goes ... 
        score = score + choose(logsoftmax, target[k, :]) * Variable(active.float())



        active *= (target[k, :] != self.v_target)
        for j in range(1 if self.no_inputs else n_examples):
            #if mode=="score":
            #    target_char_scatter = target_scatter[k, :, :]
            #elif mode=="sample":
                #target_char_scatter = Variable(self._zeros(batch_size, self.v_target+1).scatter_(1, target[k, :, None], 1))
            #for beam decoding:
            target_char_scatter = Variable(self._zeros(batch_size, self.v_target+1).scatter_(1, target[k, :, None], 1))

            decoder_states[j] = self.decoder_cell(target_char_scatter, decoder_states[j]) 
        syntax_decoder_state = self.syntax_decoder_cell(target_char_scatter, syntax_decoder_state) #TODO
        #print("DEC LEN", len(decoder_states[0]))
        return target.clone(), score.clone(), [(ds[0].clone(), ds[1].clone())  for ds in decoder_states], (syntax_decoder_state[0].clone(), syntax_decoder_state[1].clone()), active.clone()


