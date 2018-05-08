import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _use_shared_memory
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from collections import namedtuple
import torch.nn.utils.weight_norm as weightNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.autograd as autograd
import torch
from miscc.config import cfg

class STEncoder(nn.Module):
    def __init__(self, input_dim,num_layers, hidden_size, bidirection, embedding_dim):
        super(STEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.direction = 1
        if bidirection:
            self.direction = 2
        
        self.embedding_dim = embedding_dim
        
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False, bidirectional=True)
        
    def hidden_init(self, batch_size):      
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.hidden_size).cuda()),
                    autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.hidden_size).cuda()))
        return (autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.hidden_size)),
                autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.hidden_size))) 
        
    def forward(self, input, inputLenghts):
        packedInputX = pack_padded_sequence(input, inputLenghts, batch_first = False)
        inputX, self.hidden  = self.rnn(packedInputX)
        padOutputX, _ = pad_packed_sequence(inputX)
        output_ofLastHiddenState = self.hidden[0].clone()
        
        output_ofLastHiddenState = torch.mean(output_ofLastHiddenState, dim = 0)
        
        return output_ofLastHiddenState

class STDuoDecoderAttn(nn.Module):
    def __init__(self, hidden_size, embedding_dim, thought_size, vocab_size):
        super(STDuoDecoderAttn, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.thought_size = thought_size
        self.num_embeddings = vocab_size ## give current ones 
        
        #Hidden units
        hidden1 = torch.zeros(1, self.hidden_size)
        hidden2 = torch.zeros(1, self.hidden_size)
        
        if torch.cuda.is_available():
            hidden1 = hidden1.cuda()
            hidden2 = hidden2.cuda()
        
        self.hidden1 = nn.Parameter(hidden1)
        self.hidden2 = nn.Parameter(hidden2)
        
        ##LSTM cells for the decoder
        self.lstmcell_prev = nn.LSTMCell(self.thought_size+embedding_dim, self.hidden_size)
        self.lstmcell_next = nn.LSTMCell(self.thought_size+embedding_dim, self.hidden_size)
        
        self.wordProject = nn.Linear(in_features=self.hidden_size, out_features=self.num_embeddings)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.wordProject.bias.data.fill_(0)
        self.wordProject.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, inputPrev, inputNext, context):
        hidden10 = self.hidden1.expand(inputPrev.size()[1], -1).contiguous()
        hidden20 = self.hidden2.expand(inputNext.size()[1], -1).contiguous()
        
        c10 = Variable(torch.zeros(inputPrev.size(1), self.hidden_size), requires_grad=False)
        c20 = Variable(torch.zeros(inputNext.size(1), self.hidden_size), requires_grad=False)        
        
        if torch.cuda.is_available():
            c10 = Variable(torch.zeros(inputPrev.size(1), self.hidden_size).cuda(), requires_grad=False)
            c20 = Variable(torch.zeros(inputNext.size(1), self.hidden_size).cuda(), requires_grad=False)

        ## geting the context for first time from hidden unit
        #context_prev = self.attentionQuery(hidden10, keys, values)
        
        logits_prev = []
        ##concatenate the context and embedding[0]
        for i in np.arange(0, inputPrev.size()[0]):
            output_curr = torch.cat((inputPrev[i], context), 1)
            hidden10, _ = self.lstmcell_prev(output_curr, (hidden10, c10))
            
            projection_out = self.wordProject(hidden10)
            logits_prev.append(projection_out)
            #context_prev = self.attentionQuery(hidden10, keys, values)
            
            ## Project layer
        logits_prev = torch.stack(logits_prev)
        
        ## geting the context for first time from hidden unit
        #context_prev = self.attentionQuery(context, keys, values)
        
        logits_next = []
        ##concatenate the context and embedding[0]
        for i in np.arange(0, inputNext.size()[0]):
            output_curr = torch.cat((inputNext[i], context), 1)
            hidden20, _ = self.lstmcell_next(output_curr, (hidden20, c20))
            
            projection_out = self.wordProject(hidden20)
            logits_next.append(projection_out)
            #context_prev = self.attentionQuery(hidden20, keys, values)
            
            ## Project layer
        logits_next = torch.stack(logits_next)        

        return logits_prev, logits_next
    

class UniSKIP_variant(nn.Module):
    def __init__(self, encoder_model, decoder_model, embedding_dim, vocab_size,glove_model,word2idx,idx2word):
        super(UniSKIP_variant, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.glove=glove_model
        self.word2idx=word2idx
        self.idx2word=idx2word
        # self.wordembed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        # self.init_weights()
        self.linear = nn.Linear(in_features=embedding_dim, out_features=cfg.TEXT.HIDDENSTATE) 
        
    # def init_weights(self):
    #     initrange = 0.1
    #     self.wordembed.bias.data.fill_(0)
    #     self.wordembed.weight.data.uniform_(-initrange, initrange)

    def get_embeddings(self,stnce_batch):
        batch_emb=[]
        for i in stnce_batch:
            curr_sent=i.data.cpu().numpy().flatten()
            curr_sent_to_emb=[self.glove.get(self.idx2word[j])for j in curr_sent]
            batch_emb.append(np.array(curr_sent_to_emb))
        return torch.from_numpy(np.array(batch_emb))


    def forward(self, inputCurr, inputLenghts, inputPrev, inputNext):
        
        # Get the thought of the current sentence
        # print(inputCurr.data)

        # word_embed_curr = F.tanh(self.wordembed(inputCurr))

        word_embed_curr=autograd.Variable(self.get_embeddings(inputCurr),requires_grad=False)

        if torch.cuda.is_available():
            word_embed_curr=word_embed_curr.cuda()

        word_embed_curr=self.linear(word_embed_curr)

        output_ofLastHiddenState = self.encoder(word_embed_curr, inputLenghts)
        # Get the embedding for prev and next sentence 
        # word_embed_prev = F.tanh(self.wordembed(inputPrev))        
        # word_embed_next = F.tanh(self.wordembed(inputNext))

        word_embed_prev=autograd.Variable(self.get_embeddings(inputPrev), requires_grad=False)
        word_embed_next=autograd.Variable(self.get_embeddings(inputNext), requires_grad=False)

        if torch.cuda.is_available():
            word_embed_next=word_embed_next.cuda()
            word_embed_prev=word_embed_prev.cuda()

        word_embed_prev=self.linear(word_embed_prev)
        word_embed_next=self.linear(word_embed_next)

        logits_prev, logits_next = self.decoder(word_embed_prev, word_embed_next, output_ofLastHiddenState)
        
        return output_ofLastHiddenState, logits_prev, logits_next
    
## USAGE 
##num_layers, hidden_size, bidirection, embedding_dim
# encoder = STEncode(1, 512, True, 128)

# ##hidden_size, embedding_dim, thought_size, vocab_size
# decoder = STDuoDecoderAttn(256, 128, 512, 9487)

# ##encoder_model, decoder_model, embedding_dim, vocab_size
# allmodel = UniSKIP_variant(encoder, decoder, 128, 9487)