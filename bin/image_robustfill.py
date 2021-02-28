from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import copy

def choose(matrix, idxs):
    if type(idxs) is Variable: idxs = idxs.data
    assert(matrix.ndimension()==2)
    unrolled_idxs = idxs + torch.arange(0, matrix.size(0)).type_as(idxs)*matrix.size(1)
    return matrix.view(matrix.nelement())[unrolled_idxs]

class Image_RobustFill(nn.Module):
    def __init__(self, target_vocabulary, hidden_size=512, embedding_size=128, cell_type="LSTM", input_size=(3, 256, 256)):
        """
        :param: input_vocabularies: List containing a vocabulary list for each input. E.g. if learning a function f:A->B from (a,b) pairs, input_vocabularies has length 2
        :param: target_vocabulary: Vocabulary list for output
        """
        super(Image_RobustFill, self).__init__()
        self.n_encoders = 1

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_vocabularies = [None] #input_vocabularies 
        self.target_vocabulary = target_vocabulary
        self._refreshVocabularyIndex()
        self.v_inputs = None #[len(x) for x in input_vocabularies] # Number of tokens in input vocabularies
        self.v_target = len(target_vocabulary) # Number of tokens in target vocabulary

        self.no_inputs = len(self.input_vocabularies)==0

        self.cell_type=cell_type
        if cell_type=='GRU':
            self.encoder_init_h = Parameter(torch.rand(1, self.hidden_size))
            # self.encoder_cells = nn.ModuleList(
            #     [nn.GRUCell(input_size=self.v_inputs[0]+1, hidden_size=self.hidden_size, bias=True)] + 
            #     [nn.GRUCell(input_size=self.v_inputs[i]+1+self.hidden_size, hidden_size=self.hidden_size, bias=True) for i in range(1, self.n_encoders)]
            # )
            self.decoder_cell = nn.GRUCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)
        if cell_type=='LSTM':
            self.encoder_init_h = Parameter(torch.rand(1, self.hidden_size)) #Also used for decoder if self.no_inputs=True
            # self.encoder_init_cs = nn.ParameterList(
            #     [Parameter(torch.rand(1, self.hidden_size)) for i in range(len(self.v_inputs))]
            # )

            # self.encoder_cells = nn.ModuleList()
            # for i in range(self.n_encoders):
            #     input_size = self.v_inputs[i] + 1 + (self.hidden_size if i>0 else 0)
            #     self.encoder_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=self.hidden_size, bias=True))
            self.decoder_cell = nn.LSTMCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)
            self.decoder_init_c = Parameter(torch.rand(1, self.hidden_size))
        
        self.W = nn.Linear(self.hidden_size if self.no_inputs else 2*self.hidden_size, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_target+1)

        #self.As = nn.ModuleList([nn.Bilinear(self.hidden_size, self.hidden_size, 1, bias=False) for i in range(self.n_encoders)])



        #image encoder:
        self.conv1 = nn.Conv2d(3,   8, kernel_size=(3, 3),
                                padding=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(8,  16, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        #self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3),
        #                        padding=(1, 1), stride=(1, 1))
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.batch_norm2 = nn.BatchNorm2d(16)

        self.img_feat_to_embedding = nn.Sequential(nn.Linear(16*16*16, 64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64, self.hidden_size))


        #attention params:
        self.h_to_32_linear = nn.Linear(self.hidden_size, 32)
        self.img_to_32 = nn.Linear(16*16*16, 32)

        self.fc_loc = nn.Linear(32 + 32, 3 * 2)
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.img_feat_to_context = nn.Sequential(nn.Linear(16*16*16, 128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128, self.hidden_size))



    def with_target_vocabulary(self, target_vocabulary):
        """
        Returns a new network which modifies this one by changing the target vocabulary
        """
        if target_vocabulary == self.target_vocabulary:
            return self

        V_weight = []
        V_bias = []
        decoder_ih = []

        for i in range(len(target_vocabulary)):
            if target_vocabulary[i] in self.target_vocabulary:
                j = self.target_vocabulary.index(target_vocabulary[i])
                V_weight.append(self.V.weight.data[j:j+1])
                V_bias.append(self.V.bias.data[j:j+1])
                decoder_ih.append(self.decoder_cell.weight_ih.data[:,j:j+1])
            else:
                V_weight.append(self._zeros(1, self.V.weight.size(1)))
                V_bias.append(self._ones(1) * -10)
                decoder_ih.append(self._zeros(self.decoder_cell.weight_ih.data.size(0), 1))

        V_weight.append(self.V.weight.data[-1:])
        V_bias.append(self.V.bias.data[-1:])
        decoder_ih.append(self.decoder_cell.weight_ih.data[:,-1:])

        self.target_vocabulary = target_vocabulary
        self.v_target = len(target_vocabulary)

        self.V.weight.data = torch.cat(V_weight, dim=0)
        self.V.bias.data = torch.cat(V_bias, dim=0)
        self.V.out_features = self.V.bias.data.size(0)

        self.decoder_cell.weight_ih.data = torch.cat(decoder_ih, dim=1)
        self.decoder_cell.input_size = self.decoder_cell.weight_ih.data.size(1)

        self._clear_optimiser()
        self._refreshVocabularyIndex()
        return copy.deepcopy(self)

    def optimiser_step(self, batch_inputs, batch_target):
        """
        Perform a single step of SGD
        """
        if not hasattr(self, 'opt'): self._get_optimiser()
        self.opt.zero_grad()
        score = self.score(batch_inputs, batch_target, autograd=True).mean()
        (-score).backward()
        self.opt.step()
                
        return score.data.item()

    def score(self, batch_inputs, batch_target, autograd=False):
        #inputs = self._inputsToTensors(batch_inputs)
        inputs = torch.stack(tuple(torch.tensor(b) for b in batch_inputs), dim=0).float()
        if next(self.parameters()).is_cuda:
            inputs = inputs.cuda()
        inputs = [[inputs]]
        #print("INPUTS SHAPE", inputs[0][0].shape)
        target = self._targetToTensor(batch_target)
        _, score = self._run(inputs, target=target, mode="score")
        if autograd:
            return score
        else:
            return score.data

    def sample(self, batch_inputs=None, n_samples=None):
        assert batch_inputs is not None or n_samples is not None
        #inputs = self._inputsToTensors(batch_inputs)
        inputs = torch.stack(tuple(torch.tensor(b) for b in batch_inputs), dim=0).float()
        if next(self.parameters()).is_cuda:
            inputs = inputs.cuda()
        inputs = [[inputs]]

        target, score = self._run(inputs, mode="sample", n_samples=n_samples)
        target = self._tensorToOutput(target)
        return target

    def sampleAndScore(self, batch_inputs=None, n_samples=None, nRepeats=None):
        assert batch_inputs is not None or n_samples is not None
        #inputs = self._inputsToTensors(batch_inputs)
        inputs = [[batch_inputs]]
        if nRepeats is None:
            target, score = self._run(inputs, mode="sample", n_samples=n_samples)
            target = self._tensorToOutput(target)
            return target, score.data
        else:
            target = []
            score = []
            for i in range(nRepeats):
                t, s = self._run(inputs, mode="sample", n_samples=n_samples)
                t = self._tensorToOutput(t)
                target.extend(t)
                score.extend(list(s.data))
            return target, score
                                
    def _refreshVocabularyIndex(self):
        # self.input_vocabularies_index = [
        #     {self.input_vocabularies[i][j]: j for j in range(len(self.input_vocabularies[i]))}
        #     for i in range(len(self.input_vocabularies))
        # ]
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
        if next(self.parameters()).is_cuda:
            return torch.ones(*args, **kwargs).cuda()
        else:
            return torch.ones(*args, **kwargs)

    def _zeros(self, *args, **kwargs):
        if next(self.parameters()).is_cuda:
            return torch.zeros(*args, **kwargs).cuda()
        else:
            return torch.zeros(*args, **kwargs)

    def _clear_optimiser(self):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): del self.optstate

    def _get_optimiser(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
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
        super(Image_RobustFill, self).cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): self._fix_optstate()
        super(Image_RobustFill, self).cpu(*args, **kwargs)

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

    def _cell_get_h(self, cell_state):
        if self.cell_type=="GRU": return cell_state
        if self.cell_type=="LSTM": return cell_state[0]

    def _run(self, inputs, target=None, mode="sample", n_samples=None):
        """
        :param mode: "score" or "sample"
        :param list[list[LongTensor]] inputs: n_encoders * n_examples * (max length * batch_size) - change last part to batch_size * 1 x 28 x 28 or whatever it is
        :param list[LongTensor] target: max length * batch_size
        Returns output and score
        """
        assert((mode=="score" and target is not None) or mode=="sample")

        input_width = inputs[0][0].shape[-1]

        if self.no_inputs:
            batch_size = target.size(1) if mode=="score" else n_samples
        else:
            batch_size = inputs[0][0].size(0) # will reformulate this
            n_examples = len(inputs[0]) 

            #max_length_inputs = [[inputs[i][j].size(0) for j in range(n_examples)] for i in range(self.n_encoders)]
            max_length_inputs = [[3*3 for j in range(n_examples)] for i in range(self.n_encoders) ] #TODO

            # inputs_scatter = [
            #     [   Variable(self._zeros(max_length_inputs[i][j], batch_size, self.v_inputs[i]+1).scatter_(2, inputs[i][j][:, :, None], 1))
            #         for j in range(n_examples)
            #     ] for i in range(self.n_encoders)
            # ]  # n_encoders * n_examples * (max_length_input * batch_size * v_input+1)

        max_length_target = target.size(0) if target is not None else 50 #CHANGED
        score = Variable(self._zeros(batch_size))
        if target is not None: target_scatter = Variable(self._zeros(max_length_target, batch_size, self.v_target+1).scatter_(2, target[:, :, None], 1)) # max_length_target * batch_size * v_target+1

        H = [] # n_encoders * n_examples * (max_length_input * batch_size * h_encoder_size)
        embeddings = [] # n_encoders * (h for example at INPUT_EOS)
        #attention_mask = [] # n_encoders * (0 until (and including) INPUT_EOS, then -inf)

        # def attend(i, j, h):
        #     """
        #     'general' attention from https://arxiv.org/pdf/1508.04025.pdf
        #     :param i: which encoder is doing the attending (or self.n_encoders for the decoder)
        #     :param j: Index of example
        #     :param h: batch_size * hidden_size
        #     """
        #     assert(i != 0)
        #     scores = self.As[i-1](
        #         H[i-1][j].view(max_length_inputs[i-1][j] * batch_size, self.hidden_size),
        #         h.view(batch_size, self.hidden_size).repeat(max_length_inputs[i-1][j], 1)
        #     ).view(max_length_inputs[i-1][j], batch_size)
        #     c = (F.softmax(scores[:, :, None], dim=0) * H[i-1][j]).sum(0)
        #     return c

        def attend(i,j,h):
            """
            spatial transformer attn.
            H[i-1][j] should be the image itself 
            """
            assert(i != 0)
            img = H[i-1][j]
            linear_img = img.view(-1, img.size(1)*img.size(2)*img.size(3))
            theta = torch.cat((F.relu(self.h_to_32_linear(h)), F.relu(self.img_to_32(linear_img))),1) #right
            theta = self.fc_loc(theta)

            #make affine transform with 
            #sample affine grid with theta and img

            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(theta, img.size())
            transformed_img = F.grid_sample(img, grid)

            linear_transformed_img = transformed_img.view(-1, 
                transformed_img.size(1)*transformed_img.size(2)*transformed_img.size(3))

            c = self.img_feat_to_context(linear_transformed_img) # we will do b x 16 x 16 x 16 to 64 to 512
            return c

        # -------------- Image Encoders -------------
        #assume one input image:
        ii = 0

        for j in range(n_examples):
            assert j==0
            _H = []
            _embeddings = []
            num_attention = 32

            x = inputs[ii][j]

            out = F.relu(self.batch_norm1(self.conv1(x))) 
            out = F.max_pool2d(out, 2) #b x 8 x 16 x 16 
            

            out = F.relu(self.conv2(out)) #b x 16 x 16 x 16 
            out = F.max_pool2d(out, 2) 
            out = F.relu(self.conv3(out))
            if input_width == 256: out = F.max_pool2d(out, 2) 
            out = F.relu(self.batch_norm2(self.conv4(out))) #b x 16 x 16 x 16 
            if input_width == 256: out = F.max_pool2d(out, 2) 

            #out = F.max_pool2d(out, 2) #b x 128 x 7 x 7 
            #out = F.max_pool2d(out,2) #b x 256 x 3 x 3 or 4 x 4
            #out = F.relu(self.batch_norm2(self.conv4(out))) #b x 512 x 3 x 3 or 4 x 4
            #todo: make embedding (b x 512) from b x 16 x 16 x 16

            #out = out.view(out.size(0), self.hidden_size, -1) #b x 512 x (16x16) 

            #out = F.relu(self.fc1(out))
            #out = F.relu(self.fc2(out))
            #out = F.relu(self.fc3(out))

            #out = out.permute(2,0,1).contiguous()
            #out = torch.zeros(out.size()).cuda()

            _H.append(out)

            lin_out = out.view(-1, out.size(1)*out.size(2)*out.size(3))

            embedding = self.img_feat_to_embedding(lin_out) #or something like that
            _embeddings.append(embedding)

        H.append(_H)
        embeddings.append(_embeddings)


        # -------------- Encoders -------------
        # for i in range(len(self.input_vocabularies)):
        #     H.append([])
        #     embeddings.append([])
        #     attention_mask.append([])

        #     for j in range(n_examples):
        #         active = self._ones(max_length_inputs[i][j], batch_size).byte()
        #         state = self._encoder_get_init(i, batch_size=batch_size, h=embeddings[i-1][j] if i>0 else None)
        #         hs = []
        #         h = self._cell_get_h(state)
        #         for k in range(max_length_inputs[i][j]):
        #             if i==0:
        #                 state = self.encoder_cells[i](inputs_scatter[i][j][k, :, :], state)
        #             else:
        #                 state = self.encoder_cells[i](torch.cat([inputs_scatter[i][j][k, :, :], attend(i, j, h)], 1), state)
        #             if k+1 < max_length_inputs[i][j]: active[k+1, :] = active[k, :] * (inputs[i][j][k, :] != self.v_inputs[i])
        #             h = self._cell_get_h(state) 
        #             hs.append(h[None, :, :])
        #         H[i].append(torch.cat(hs, 0))
        #         embedding_idx = active.sum(0).long() - 1
        #         embedding = H[i][j].gather(0, Variable(embedding_idx[None, :, None].repeat(1, 1, self.hidden_size)))[0]
        #         embeddings[i].append(embedding)
        #         #embedding.size() == batchsize x hidden_size
        #         attention_mask[i].append(Variable(active.float().log()))


        # ------------------ Decoder -----------------
        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        target = target if mode=="score" else self._zeros(max_length_target, batch_size).long()
        if self.no_inputs: decoder_states = [self._decoder_get_init(batch_size=batch_size)]
        else: decoder_states = [self._decoder_get_init(embeddings[self.n_encoders-1][j]) for j in range(n_examples)] #P
        active = self._ones(batch_size).byte()
        for k in range(max_length_target):
            FC = []
            for j in range(1 if self.no_inputs else n_examples):
                h = self._cell_get_h(decoder_states[j])
                p_aug = h if self.no_inputs else torch.cat([h, attend(self.n_encoders, j, h)], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))
            m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
            logsoftmax = F.log_softmax(self.V(m), dim=1)
            if mode=="sample": target[k, :] = torch.multinomial(logsoftmax.data.exp(), 1)[:, 0]
            score = score + choose(logsoftmax, target[k, :]) * Variable(active.float())
            active *= (target[k, :] != self.v_target)
            for j in range(1 if self.no_inputs else n_examples):
                if mode=="score":
                    target_char_scatter = target_scatter[k, :, :]
                elif mode=="sample":
                    target_char_scatter = Variable(self._zeros(batch_size, self.v_target+1).scatter_(1, target[k, :, None], 1))
                decoder_states[j] = self.decoder_cell(target_char_scatter, decoder_states[j]) 
        return target, score

    def _inputsToTensors(self, inputsss):
        """
        :param inputs: size = nBatch * nExamples * nEncoders (or nBatch*nExamples is n_encoders=1)
        Returns nEncoders * nExamples tensors of size nBatch * max_len
        """
        #print("WARNING: you have hit a depricated function, _inputsToTensors")

        # if self.n_encoders == 0: return []
        # tensors = []
        # for i in range(self.n_encoders):
        #     tensors.append([])
        #     for j in range(len(inputsss[0])):
        #         if self.n_encoders == 1: inputs = [x[j] for x in inputsss]
        #         else: inputs = [x[j][i] for x in inputsss]

        #         maxlen = max(len(s) for s in inputs)
        #         t = self._ones(maxlen+1, len(inputs)).long()*self.v_inputs[i]
        #         for k in range(len(inputs)):
        #             s = inputs[k]
        #             if len(s)>0: t[:len(s), k] = torch.LongTensor([self.input_vocabularies_index[i][x] for x in s])
        #         tensors[i].append(t)



        #assert inputsss.shape == bx500
        return [inputsss]

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
        :param tensor: max_length * batch_size
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


if __name__ == '__main__':
    from torchvision import datasets, transforms, utils
    import time

    batch_size = 32
    max_length = 15

    rescaling     = lambda x : (x - .5) * 2.
    rescaling_inv = lambda x : .5 * x  + .5
    flip = lambda x : - x
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
    resizing = lambda x: x.resize((28,28))
    omni_transforms = transforms.Compose([resizing, transforms.ToTensor(), rescaling, flip]) #TODO: check this, but i think i don't want rescaling

    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
    train_loader = torch.utils.data.DataLoader(datasets.Omniglot('../vhe/data', download=True, 
                        background=True, transform=omni_transforms), batch_size=batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(datasets.Omniglot('../vhe/data', download=True, 
                        background=False, transform=omni_transforms), batch_size=batch_size, 
                            shuffle=True, **kwargs)  


    vocab = [str(i) for i in range(10)]

    model = Image_RobustFill(input_vocabularies=None, target_vocabulary=vocab)
    model.cuda()

    print("Training:")
    start=time.time()

    for i, batch in enumerate(train_loader):
        xs, indx = batch
        targets = [ [ char for char in str(ind.numpy())] for ind in indx]
        score = model.optimiser_step(xs.cuda(), targets)
        if i%10==0: print("Iteration %d" % (i), "Score %3.3f" % score, "(%3.3f seconds per iteration)" % ((time.time()-start)/(i+1)))
