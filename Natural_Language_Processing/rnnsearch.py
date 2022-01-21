import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self, n_inputs, n_embeddings, n_hiddens):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.embedding = nn.Embedding(n_inputs, n_embeddings)
        self.bidirectional_gru = nn.GRU(n_embeddings, n_hiddens, bidirectional=True)
        self.fc = nn.Linear(n_hiddens * 2, n_hiddens)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.bidirectional_gru(x)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = torch.tanh(self.fc(hidden))
        return output, hidden


class Alignment(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.v = nn.Parameter(nn.init.uniform_(torch.empty(n_hiddens)))
        self.align = nn.Linear(self.n_hiddens * 3, self.n_hiddens)
        
    def forward(self, h , s):
        e = torch.cat([h, s], dim = 2)
        e = torch.tanh(self.align(e))
        e = e.transpose(1, 2)
        v = self.v.repeat(s.size(0), 1).unsqueeze(1)
        e = torch.bmm(v, e)
        return e.squeeze(1)

class Attention(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.align = Alignment(self.n_hiddens)
    
    def forward(self, h, s):
        time_step = s.shape[0]
        h = h.unsqueeze(1)
        h = h.repeat(1, time_step, 1)
        s = s.permute(1, 0, 2)
        energy = self.align(h, s)
        return F.softmax(energy, dim=1).unsqueeze(1)


class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m
        


class Decoder(nn.Module):
    def __init__(self, n_outputs, n_embeddings, n_hiddens, n_maxout):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.embedding = nn.Embedding(n_outputs, n_embeddings)
        self.attention_layer = Attention(self.n_hiddens)
        self.gru = nn.GRU(n_embeddings + n_hiddens * 2, n_hiddens)

        self.maxout = Maxout(n_hiddens * 3 + n_embeddings, n_maxout, 2)
        self.out = nn.Linear(n_maxout, n_outputs)

    def forward(self, input, h, s):
        embedded = self.embedding(input)
        attention = self.attention_layer(h, s)
        context = attention.bmm(s.transpose(0, 1)).transpose(0, 1)
        embedded = embedded.unsqueeze(0)
        input = torch.cat([embedded, context], 2)
        h = h.unsqueeze(0)
        out, hidden = self.gru(input, h)
        maxout_input = torch.cat([h, embedded, context], dim=2)
        out = self.maxout(maxout_input).squeeze(0)
        out = self.out(out)
        out = F.log_softmax(out, dim=1)
        return out, hidden.squeeze(0)


class RNNsearch(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_embeddings, n_hiddens, n_maxout, device):
        super().__init__()
        self.n_outputs = n_outputs
        self.device = device

        self.encoder = Encoder(n_inputs, n_embeddings, n_hiddens)
        self.decoder = Decoder(n_outputs, n_embeddings, n_hiddens, n_maxout)

        
    def forward(self, x, target, teacher_forcing_ratio):
        
        encoder_outputs, hiddens = self.encoder(x)

        output = target[0, :]

        outputs = torch.zeros(target.shape[0], target.shape[1], self.n_outputs).to(self.device)

        for t in range(1, target.shape[0]):
            output, hiddens = self.decoder(output, hiddens, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            output = target[t] if teacher_force else output.argmax(1)
            
    
        return outputs


def inference(model, src_vocab, trg_vocab, src_tokenizer, srcs, device):
    tokens = []
    for src in srcs:
        tokens.append([src_vocab[s] for s in src_tokenizer(src)])
    
    tokens = torch.LongTensor(tokens).cuda().transpose(0, 1)
    v = list(trg_vocab.get_stoi().values())
    k = list(trg_vocab.get_stoi().keys())
    trg_dict = {}
    for i in tqdm(range(len(trg_vocab.get_stoi()))):
        trg_dict[v[i]] = k[i]
    model.eval()

    with torch.no_grad():
        out = model(tokens, torch.LongTensor([[t for t in range(50)] for t in range(1)]).transpose(0, 1).cuda(), 0).detach().cpu()
        out = out.transpose(0, 1)

    res = []
    for o in out:
        res.append(' '.join([trg_dict[t.item()] for t in F.softmax(o, dim=1).argmax(1) if not trg_dict[t.item()] == '<eos>']))
    
    return res