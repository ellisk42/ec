import random

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optimization
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

class LineEncoder(nn.Module):
    def __init__(self, lexicon, H=256):
        super(LineEncoder, self).__init__()

        self.encoder = nn.Embedding(len(lexicon), H)
        self.lexicon = lexicon
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
    def forward(self, objects):
        return self.encoder(torch.tensor([self.wordToIndex[o] for o in objects]))
        

class LineDecoder(nn.Module):
    def __init__(self, lexicon, H=256, layers=1):
        super(LineDecoder, self).__init__()

        self.model = nn.GRU(H, H, layers)

        self.specialSymbols = [
            "STARTING", "ENDING", "POINTER"
            ]

        self.lexicon = lexicon + self.specialSymbols
        self.wordToIndex = {w: j for j,w in enumerate(self.lexicon) }
        self.embedding = nn.Embedding(len(self.lexicon), H)

        self.output = nn.Sequential(nn.Linear(H, len(self.lexicon)),
                                    nn.LogSoftmax())
        
        self.decoderToPointer = nn.Linear(H, H)
        self.encoderToPointer = nn.Linear(H, H)
        self.attentionSelector = nn.Linear(H, 1)

        self.pointerIndex = self.wordToIndex["POINTER"]

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        hidden = hidden.unsqueeze(0)        
        output, hidden = self.model(input, hidden)
        input = input.squeeze(0)
        output = output.squeeze(0)
        return self.output(output), hidden

    def pointerAttention(self, hiddenStates, objectEncodings):
        hiddenStates = self.decoderToPointer(hiddenStates)
        objectEncodings = self.encoderToPointer(objectEncodings)

        _h = hiddenStates.unsqueeze(1).repeat(1, objectEncodings.size(0), 1)
        _o = objectEncodings.unsqueeze(0).repeat(hiddenStates.size(0), 1, 1)
        attention = self.attentionSelector(torch.tanh(_h + _o))
        return F.log_softmax(attention.squeeze(2), dim=1)        

    def logLikelihood(self, initialState, target, encodedInputs):
        symbolSequence = [self.wordToIndex[t if isinstance(t,str) else "POINTER"]
                          for t in ["STARTING"] + target + ["ENDING"] ]
        
        # inputSequence : L x B x H
        inputSequence = self.embedding(torch.tensor(symbolSequence[:-1])).unsqueeze(1)
        outputSequence = torch.tensor(symbolSequence[1:])

        h0 = initialState.unsqueeze(0).unsqueeze(0)

        o, h = self.model(inputSequence, h0)

        # output sequence log likelihood, ignoring pointer values
        sll = -F.nll_loss(self.output(o.squeeze(1)), outputSequence, reduce=True, size_average=False)

        # pointer value log likelihood
        pointerTimes = [t - 1 for t,s in enumerate(symbolSequence) if self.pointerIndex == s ]
        pointerValues = [v for v in target if isinstance(v, int) ]
        pointerHiddens = o[torch.tensor(pointerTimes),:,:].squeeze(1)
        
        attention = self.pointerAttention(pointerHiddens, encodedInputs)
        pll = -F.nll_loss(attention, torch.tensor(pointerValues),
                          reduce=True, size_average=False)
        return sll + pll

    def sample(self, initialState, encodedInputs):
        sequence = ["STARTING"]
        h = initialState
        while len(sequence) < 100:
            lastWord = sequence[-1]
            if isinstance(lastWord,int): lastWord = "POINTER"
            i = self.embedding(torch.tensor(self.wordToIndex[lastWord]))
            o,h = self.model(i.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0))
            o = o.squeeze(0).squeeze(0)
            h = h.squeeze(0).squeeze(0)

            # Sample the next symbol
            distribution = self.output(o)
            next_symbol = self.lexicon[torch.multinomial(distribution.exp(), 1)[0].data.item()]
            if next_symbol == "ENDING":
                break
            if next_symbol == "POINTER":
                # Sample the next pointer
                a = self.pointerAttention(h.unsqueeze(0), encodedInputs).squeeze(0)
                next_symbol = torch.multinomial(a.exp(),1)[0].data.item()                

            sequence.append(next_symbol)
                
            
        return sequence
            

            
            
class PointerNetwork(nn.Module):
    def __init__(self, lexicon, H=256):
        super(PointerNetwork, self).__init__()
        self.encoder = LineEncoder(lexicon, H=H)
        self.decoder = LineDecoder(lexicon, H=H)

    def gradientStep(self, optimizer, inputObjects, outputSequence,
                     verbose=False):
        self.zero_grad()
        l = -self.decoder.logLikelihood(torch.zeros(256), outputSequence,
                                        self.encoder(inputObjects))
        l.backward()
        optimizer.step()
        if verbose:
            print("loss",l.data.item())

    def sample(self, inputObjects):
        return [ inputObjects[s] if isinstance(s,int) else s
                 for s in self.decoder.sample(torch.zeros(256),
                                              self.encoder(inputObjects))         ]
        
        
        
m = PointerNetwork(["large","small"] + [str(n) for n in range(10) ])
optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
for n in range(90000):
    x = str(random.choice(range(10)))
    y = str(random.choice(range(10)))
    if x == y: continue
    large = max(x,y)
    small = min(x,y)
    if random.choice([False,True]):
        sequence = ["large", int(large == y), int(large == y),
                    "small", int(small == y)]
    else:
        sequence = ["small", int(small == y),
                    "large", int(large == y)]
    verbose = n%50 == 0
    m.gradientStep(optimizer, [x,y], sequence, verbose=verbose)
    if verbose:
        print([x,y],"goes to",m.sample([x,y]))
