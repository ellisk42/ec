"""
Deprecated network.py module. This file only exists to support backwards-compatibility
with old pickle files. See lib/__init__.py for more information.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


# UPGRADING TO INPUT -> OUTPUT -> TARGET
# Todo:
# [X] Output attending to input
# [X] Target attending to output
# [ ] check passing hidden state between encoders/decoder (+ pass c?)
# [ ] add v_output


def choose(matrix, idxs):
    if isinstance(idxs, Variable):
        idxs = idxs.data
    assert(matrix.ndimension() == 2)
    unrolled_idxs = idxs + \
        torch.arange(0, matrix.size(0)).type_as(idxs) * matrix.size(1)
    return matrix.view(matrix.nelement())[unrolled_idxs]


class Network(nn.Module):
    """
    Todo:
    - Beam search
    - check if this is right? attend during P->FC rather than during softmax->P?
    - allow length 0 inputs/targets
    - give n_examples as input to FC
    - Initialise new weights randomly, rather than as zeroes
    """

    def __init__(
            self,
            input_vocabulary,
            target_vocabulary,
            hidden_size=512,
            embedding_size=128,
            cell_type="LSTM"):
        """
        :param list input_vocabulary: list of possible inputs
        :param list target_vocabulary: list of possible targets
        """
        super(Network, self).__init__()
        self.h_input_encoder_size = hidden_size
        self.h_output_encoder_size = hidden_size
        self.h_decoder_size = hidden_size
        self.embedding_size = embedding_size
        self.input_vocabulary = input_vocabulary
        self.target_vocabulary = target_vocabulary
        # Number of tokens in input vocabulary
        self.v_input = len(input_vocabulary)
        # Number of tokens in target vocabulary
        self.v_target = len(target_vocabulary)

        self.cell_type = cell_type
        if cell_type == 'GRU':
            self.input_encoder_cell = nn.GRUCell(
                input_size=self.v_input + 1,
                hidden_size=self.h_input_encoder_size,
                bias=True)
            self.input_encoder_init = Parameter(
                torch.rand(1, self.h_input_encoder_size))
            self.output_encoder_cell = nn.GRUCell(
                input_size=self.v_input +
                1 +
                self.h_input_encoder_size,
                hidden_size=self.h_output_encoder_size,
                bias=True)
            self.decoder_cell = nn.GRUCell(
                input_size=self.v_target + 1,
                hidden_size=self.h_decoder_size,
                bias=True)
        if cell_type == 'LSTM':
            self.input_encoder_cell = nn.LSTMCell(
                input_size=self.v_input + 1,
                hidden_size=self.h_input_encoder_size,
                bias=True)
            self.input_encoder_init = nn.ParameterList([Parameter(torch.rand(
                1, self.h_input_encoder_size)), Parameter(torch.rand(1, self.h_input_encoder_size))])
            self.output_encoder_cell = nn.LSTMCell(
                input_size=self.v_input +
                1 +
                self.h_input_encoder_size,
                hidden_size=self.h_output_encoder_size,
                bias=True)
            self.output_encoder_init_c = Parameter(
                torch.rand(1, self.h_output_encoder_size))
            self.decoder_cell = nn.LSTMCell(
                input_size=self.v_target + 1,
                hidden_size=self.h_decoder_size,
                bias=True)
            self.decoder_init_c = Parameter(torch.rand(1, self.h_decoder_size))

        self.W = nn.Linear(
            self.h_output_encoder_size +
            self.h_decoder_size,
            self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_target + 1)
        self.input_A = nn.Bilinear(
            self.h_input_encoder_size,
            self.h_output_encoder_size,
            1,
            bias=False)
        self.output_A = nn.Bilinear(
            self.h_output_encoder_size,
            self.h_decoder_size,
            1,
            bias=False)
        self.input_EOS = torch.zeros(1, self.v_input + 1)
        self.input_EOS[:, -1] = 1
        self.input_EOS = Parameter(self.input_EOS)
        self.output_EOS = torch.zeros(1, self.v_input + 1)
        self.output_EOS[:, -1] = 1
        self.output_EOS = Parameter(self.output_EOS)
        self.target_EOS = torch.zeros(1, self.v_target + 1)
        self.target_EOS[:, -1] = 1
        self.target_EOS = Parameter(self.target_EOS)

    def __getstate__(self):
        if hasattr(self, 'opt'):
            return dict([(k, v) for k, v in self.__dict__.items(
            ) if k is not 'opt'] + [('optstate', self.opt.state_dict())])
            # return {**{k:v for k,v in self.__dict__.items() if k is not 'opt'},
            #         'optstate': self.opt.state_dict()}
        else:
            return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Legacy:
        if isinstance(self.input_encoder_init, tuple):
            self.input_encoder_init = nn.ParameterList(
                list(self.input_encoder_init))

    def clear_optimiser(self):
        if hasattr(self, 'opt'):
            del self.opt
        if hasattr(self, 'optstate'):
            del self.optstate

    def get_optimiser(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        if hasattr(self, 'optstate'):
            self.opt.load_state_dict(self.optstate)

    def optimiser_step(self, inputs, outputs, target):
        if not hasattr(self, 'opt'):
            self.get_optimiser()
        score = self.score(inputs, outputs, target, autograd=True).mean()
        (-score).backward()
        self.opt.step()
        self.opt.zero_grad()
        return score.data[0]

    def set_target_vocabulary(self, target_vocabulary):
        if target_vocabulary == self.target_vocabulary:
            return

        V_weight = []
        V_bias = []
        decoder_ih = []

        for i in range(len(target_vocabulary)):
            if target_vocabulary[i] in self.target_vocabulary:
                j = self.target_vocabulary.index(target_vocabulary[i])
                V_weight.append(self.V.weight.data[j:j + 1])
                V_bias.append(self.V.bias.data[j:j + 1])
                decoder_ih.append(self.decoder_cell.weight_ih.data[:, j:j + 1])
            else:
                V_weight.append(torch.zeros(1, self.V.weight.size(1)))
                V_bias.append(torch.ones(1) * -10)
                decoder_ih.append(
                    torch.zeros(
                        self.decoder_cell.weight_ih.data.size(0), 1))

        V_weight.append(self.V.weight.data[-1:])
        V_bias.append(self.V.bias.data[-1:])
        decoder_ih.append(self.decoder_cell.weight_ih.data[:, -1:])

        self.target_vocabulary = target_vocabulary
        self.v_target = len(target_vocabulary)
        self.target_EOS.data = torch.zeros(1, self.v_target + 1)
        self.target_EOS.data[:, -1] = 1

        self.V.weight.data = torch.cat(V_weight, dim=0)
        self.V.bias.data = torch.cat(V_bias, dim=0)
        self.V.out_features = self.V.bias.data.size(0)

        self.decoder_cell.weight_ih.data = torch.cat(decoder_ih, dim=1)
        self.decoder_cell.input_size = self.decoder_cell.weight_ih.data.size(1)

        self.clear_optimiser()

    def input_encoder_get_init(self, batch_size):
        if self.cell_type == "GRU":
            return self.input_encoder_init.repeat(batch_size, 1)
        if self.cell_type == "LSTM":
            return tuple(x.repeat(batch_size, 1)
                         for x in self.input_encoder_init)

    def output_encoder_get_init(self, input_encoder_h):
        if self.cell_type == "GRU":
            return input_encoder_h
        if self.cell_type == "LSTM":
            return (
                input_encoder_h,
                self.output_encoder_init_c.repeat(
                    input_encoder_h.size(0),
                    1))

    def decoder_get_init(self, output_encoder_h):
        if self.cell_type == "GRU":
            return output_encoder_h
        if self.cell_type == "LSTM":
            return (
                output_encoder_h,
                self.decoder_init_c.repeat(
                    output_encoder_h.size(0),
                    1))

    def cell_get_h(self, cell_state):
        if self.cell_type == "GRU":
            return cell_state
        if self.cell_type == "LSTM":
            return cell_state[0]

    def score(self, inputs, outputs, target, autograd=False):
        inputs = self.inputsToTensors(inputs)
        outputs = self.inputsToTensors(outputs)
        target = self.targetToTensor(target)
        target, score = self.run(inputs, outputs, target=target, mode="score")
        # target = self.tensorToOutput(target)
        if autograd:
            return score
        else:
            return score.data

    def sample(self, inputs, outputs):
        inputs = self.inputsToTensors(inputs)
        outputs = self.inputsToTensors(outputs)
        target, score = self.run(inputs, outputs, mode="sample")
        target = self.tensorToOutput(target)
        return target

    def sampleAndScore(self, inputs, outputs, nRepeats=None):
        inputs = self.inputsToTensors(inputs)
        outputs = self.inputsToTensors(outputs)
        if nRepeats is None:
            target, score = self.run(inputs, outputs, mode="sample")
            target = self.tensorToOutput(target)
            return target, score.data
        else:
            target = []
            score = []
            for i in range(nRepeats):
                # print("repeat %d" % i)
                t, s = self.run(inputs, outputs, mode="sample")
                t = self.tensorToOutput(t)
                target.extend(t)
                score.extend(list(s.data))
            return target, score

    def run(self, inputs, outputs, target=None, mode="sample"):
        """
        :param mode: "score" returns log p(target|input), "sample" returns target ~ p(-|input)
        :param List[LongTensor] inputs: n_examples * (max_length_input * batch_size)
        :param List[LongTensor] target: max_length_target * batch_size
        """
        assert((mode == "score" and target is not None) or mode == "sample")

        n_examples = len(inputs)
        max_length_input = [inputs[j].size(0) for j in range(n_examples)]
        max_length_output = [outputs[j].size(0) for j in range(n_examples)]
        max_length_target = target.size(0) if target is not None else 10
        batch_size = inputs[0].size(1)

        score = Variable(torch.zeros(batch_size))
        inputs_scatter = [Variable(torch.zeros(max_length_input[j], batch_size, self.v_input + 1).scatter_(
            2, inputs[j][:, :, None], 1)) for j in range(n_examples)]  # n_examples * (max_length_input * batch_size * v_input+1)
        outputs_scatter = [Variable(torch.zeros(max_length_output[j], batch_size, self.v_input + 1).scatter_(
            2, outputs[j][:, :, None], 1)) for j in range(n_examples)]  # n_examples * (max_length_output * batch_size * v_input+1)
        if target is not None:
            target_scatter = Variable(torch.zeros(max_length_target,
                                                  batch_size,
                                                  self.v_target + 1).scatter_(2,
                                                                              target[:,
                                                                                     :,
                                                                                     None],
                                                                              1))  # max_length_target * batch_size * v_target+1

        # -------------- Input Encoder -------------

        # n_examples * (max_length_input * batch_size * h_encoder_size)
        input_H = []
        input_embeddings = []  # h for example at INPUT_EOS
        # 0 until (and including) INPUT_EOS, then -inf
        input_attention_mask = []
        for j in range(n_examples):
            active = torch.Tensor(max_length_input[j], batch_size).byte()
            active[0, :] = 1
            state = self.input_encoder_get_init(batch_size)
            hs = []
            for i in range(max_length_input[j]):
                state = self.input_encoder_cell(
                    inputs_scatter[j][i, :, :], state)
                if i + 1 < max_length_input[j]:
                    active[i + 1, :] = active[i, :] * \
                        (inputs[j][i, :] != self.v_input)
                h = self.cell_get_h(state)
                hs.append(h[None, :, :])
            input_H.append(torch.cat(hs, 0))
            embedding_idx = active.sum(0).long() - 1
            embedding = input_H[j].gather(0, Variable(
                embedding_idx[None, :, None].repeat(1, 1, self.h_input_encoder_size)))[0]
            input_embeddings.append(embedding)
            input_attention_mask.append(Variable(active.float().log()))

        # -------------- Output Encoder -------------

        def input_attend(j, h_out):
            """
            'general' attention from https://arxiv.org/pdf/1508.04025.pdf
            :param j: Index of example
            :param h_out: batch_size * h_output_encoder_size
            """
            scores = self.input_A(
                input_H[j].view(
                    max_length_input[j] * batch_size,
                    self.h_input_encoder_size),
                h_out.view(
                    batch_size,
                    self.h_output_encoder_size).repeat(
                    max_length_input[j],
                    1)).view(
                max_length_input[j],
                batch_size) + input_attention_mask[j]
            c = (F.softmax(scores[:, :, None], dim=0) * input_H[j]).sum(0)
            return c

        # n_examples * (max_length_input * batch_size * h_encoder_size)
        output_H = []
        output_embeddings = []  # h for example at INPUT_EOS
        # 0 until (and including) INPUT_EOS, then -inf
        output_attention_mask = []
        for j in range(n_examples):
            active = torch.Tensor(max_length_output[j], batch_size).byte()
            active[0, :] = 1
            state = self.output_encoder_get_init(input_embeddings[j])
            hs = []
            h = self.cell_get_h(state)
            for i in range(max_length_output[j]):
                state = self.output_encoder_cell(torch.cat(
                    [outputs_scatter[j][i, :, :], input_attend(j, h)], 1), state)
                if i + 1 < max_length_output[j]:
                    active[i + 1, :] = active[i, :] * \
                        (outputs[j][i, :] != self.v_input)
                h = self.cell_get_h(state)
                hs.append(h[None, :, :])
            output_H.append(torch.cat(hs, 0))
            embedding_idx = active.sum(0).long() - 1
            embedding = output_H[j].gather(0, Variable(
                embedding_idx[None, :, None].repeat(1, 1, self.h_output_encoder_size)))[0]
            output_embeddings.append(embedding)
            output_attention_mask.append(Variable(active.float().log()))

        # ------------------ Decoder -----------------

        def output_attend(j, h_dec):
            """
            'general' attention from https://arxiv.org/pdf/1508.04025.pdf
            :param j: Index of example
            :param h_dec: batch_size * h_decoder_size
            """
            scores = self.output_A(
                output_H[j].view(
                    max_length_output[j] * batch_size,
                    self.h_output_encoder_size),
                h_dec.view(
                    batch_size,
                    self.h_decoder_size).repeat(
                    max_length_output[j],
                    1)).view(
                max_length_output[j],
                batch_size) + output_attention_mask[j]
            c = (F.softmax(scores[:, :, None], dim=0) * output_H[j]).sum(0)
            return c

        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        target = target if mode == "score" else torch.zeros(
            max_length_target, batch_size).long()
        decoder_states = [
            self.decoder_get_init(
                output_embeddings[j]) for j in range(n_examples)]  # P
        active = torch.ones(batch_size).byte()
        for i in range(max_length_target):
            FC = []
            for j in range(n_examples):
                h = self.cell_get_h(decoder_states[j])
                p_aug = torch.cat([h, output_attend(j, h)], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))
            # batch_size * embedding_size
            m = torch.max(torch.cat(FC, 0), 0)[0]
            logsoftmax = F.log_softmax(self.V(m), dim=1)
            if mode == "sample":
                target[i, :] = torch.multinomial(
                    logsoftmax.data.exp(), 1)[:, 0]
            score = score + \
                choose(logsoftmax, target[i, :]) * Variable(active.float())
            active *= (target[i, :] != self.v_target)
            for j in range(n_examples):
                if mode == "score":
                    target_char_scatter = target_scatter[i, :, :]
                elif mode == "sample":
                    target_char_scatter = Variable(torch.zeros(
                        batch_size, self.v_target + 1).scatter_(1, target[i, :, None], 1))
                decoder_states[j] = self.decoder_cell(
                    target_char_scatter, decoder_states[j])
        return target, score

    def inputsToTensors(self, inputss):
        """
        :param inputss: size = nBatch * nExamples
        """
        tensors = []
        for j in range(len(inputss[0])):
            inputs = [x[j] for x in inputss]
            maxlen = max(len(s) for s in inputs)
            t = torch.ones(
                1 if maxlen == 0 else maxlen + 1,
                len(inputs)).long() * self.v_input
            for i in range(len(inputs)):
                s = inputs[i]
                if len(s) > 0:
                    t[:len(s), i] = torch.LongTensor(
                        [self.input_vocabulary.index(x) for x in s])
            tensors.append(t)
        return tensors

    def targetToTensor(self, targets):
        """
        :param targets:
        """
        maxlen = max(len(s) for s in targets)
        t = torch.ones(
            1 if maxlen == 0 else maxlen + 1,
            len(targets)).long() * self.v_target
        for i in range(len(targets)):
            s = targets[i]
            if len(s) > 0:
                t[:len(s), i] = torch.LongTensor(
                    [self.target_vocabulary.index(x) for x in s])
        return t

    def tensorToOutput(self, tensor):
        """
        :param tensor: max_length * batch_size
        """
        out = []
        for i in range(tensor.size(1)):
            l = tensor[:, i].tolist()
            if l[0] == self.v_target:
                out.append([])
            elif self.v_target in l:
                final = tensor[:, i].tolist().index(self.v_target)
                out.append([self.target_vocabulary[x]
                            for x in tensor[:final, i]])
            else:
                out.append([self.target_vocabulary[x] for x in tensor[:, i]])
        return out
