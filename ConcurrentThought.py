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

class CaptionLoader():
    """ Load the WSJ speech dataset with transcripts.
        
        Ensure WSJ_PATH is path to directory containing 
        all data files (.npy) provided on Kaggle.
        
        Example usage:
            loader = WSJ()
            trainX, trainY = loader.train
            assert(trainX.shape[0] == 24590)
            
    """

    os.environ['WSJ_PATH'] = ""
    
    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_raw(os.environ['WSJ_PATH'], 'dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_raw(os.environ['WSJ_PATH'], 'train')
        return self.train_set
  
    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (np.load(os.path.join(os.environ['WSJ_PATH'], 'test.npy'), encoding='bytes'), None)
        return self.test_set
    
def load_raw(path, name):
    return (
        np.load(os.path.join(path, 'BaLSaKGAN-/skip_thought/devX.npy'), encoding='bytes'), 
        np.load(os.path.join(path, 'BaLSaKGAN-/skip_thought/devX.npy'), encoding='bytes')
    )


captionload = CaptionLoader()
trainX, trainX = captionload.dev

class CaptionDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X

    def __getitem__(self, index):
        cap = np.random.randint(0,len(self.X[index]))
        return self.X[index][(cap-1)%len(self.X[index])], self.X[index][cap], self.X[index][(cap+1)%len(self.X[index])]

    def __len__(self):
        return self.X.shape[0]
    
dataCaptrain = CaptionDataset(trainX)

def collate_fn(data):
    
    ## Sorting based on the utternce lenght
    # middle sentence is the one required to send into the encoder to produce the thoughts
    data.sort(key=lambda x: x[1].shape[0], reverse=True)
    
    ## Unzip the data
    X, Y, Z = zip(*data)
    
    ## Now padding the X
    Xlenghts = [Sample.shape[0] for Sample in X]
    Ylenghts = [Sample.shape[0] for Sample in Y]
    Zlenghts = [Sample.shape[0] for Sample in Z]
    
    if _use_shared_memory:
        paddedArrayX = torch.FloatStorage._new_shared(max(Xlenghts)*len(Xlenghts)).new(max(Xlenghts),len(Xlenghts)).zero_()
        maskArrayX = torch.FloatStorage._new_shared(max(Xlenghts)*len(Xlenghts)).new(max(Xlenghts),len(Xlenghts)).zero_()
        
        paddedArrayY = torch.FloatStorage._new_shared(max(Ylenghts)*len(Ylenghts)).new(max(Ylenghts),len(Ylenghts)).zero_()
        
        paddedArrayZ = torch.FloatStorage._new_shared(max(Zlenghts)*len(Zlenghts)).new(max(Zlenghts),len(Zlenghts)).zero_()
        maskArrayZ = torch.FloatStorage._new_shared(max(Zlenghts)*len(Zlenghts)).new(max(Zlenghts),len(Zlenghts)).zero_()
    else:
        paddedArrayX = torch.FloatTensor(max(Xlenghts),len(Xlenghts)).zero_()
        maskArrayX = torch.FloatTensor(max(Xlenghts),len(Xlenghts)).zero_()
        
        paddedArrayY = torch.FloatTensor(max(Ylenghts),len(Ylenghts)).zero_()
        
        paddedArrayZ = torch.FloatTensor(max(Zlenghts),len(Zlenghts)).zero_()
        maskArrayZ = torch.FloatTensor(max(Zlenghts),len(Zlenghts)).zero_()
        
    for idx, Sample in enumerate(X):
        paddedArrayX[:Sample.shape[0], idx] = torch.from_numpy(Sample)
        maskArrayX[:Sample.shape[0], idx] = 1
        
    for idx, Sample in enumerate(Y):
        paddedArrayY[:Sample.shape[0], idx] = torch.from_numpy(Sample)
        
    for idx, Sample in enumerate(Z):
        paddedArrayZ[:Sample.shape[0], idx] = torch.from_numpy(Sample)
        maskArrayZ[:Sample.shape[0], idx] = 1

    return paddedArrayX, maskArrayX, paddedArrayY, Ylenghts, paddedArrayZ, maskArrayZ

TrainLoader = DataLoader(dataCaptrain,
                          batch_size=32,
                          collate_fn=collate_fn,
                          shuffle=True,
                          num_workers = 3,
                          pin_memory=True # CUDA only
                          )

class STEncoder(nn.Module):
    def __init__(self):
        super(STEncoder, self).__init__()
        
        self.num_layers = 1
        self.thought_size = 512
        self.direction = 2
        self.embedding_dim = 128
        
        self.rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.thought_size, num_layers=self.num_layers, batch_first=False, bidirectional=True)
        
    def hidden_init(self, batch_size):      
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.thought_size).cuda()),
                    autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.thought_size).cuda()))
        return (autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.thought_size)),
                autograd.Variable(torch.zeros(self.direction*self.num_layers, batch_size, self.thought_size))) 
        
    def forward(self, input, inputLenghts):
        packedInputX = pack_padded_sequence(input, inputLenghts, batch_first = False)
        inputX, self.hidden  = self.rnn(packedInputX)
        padOutputX, _ = pad_packed_sequence(inputX)
        output_ofLastHiddenState = self.hidden[0].clone()
        
        output_ofLastHiddenState = torch.mean(output_ofLastHiddenState, dim = 0)
        
        return output_ofLastHiddenState

class STDuoDecoderAttn(nn.Module):
    def __init__(self, vocab_size):
        super(STDuoDecoderAttn, self).__init__()
        self.hidden_size = 256
        self.embedding_dim = 128
        self.num_embeddings = vocab_size ## give current ones 
        self.key_space = 128
        
        #Hidden units
        hidden1 = torch.zeros(1, 256)
        hidden2 = torch.zeros(1, 256)
        if torch.cuda.is_available():
            hidden1 = hidden1.cuda()
            hidden2 = hidden2.cuda()
        
        self.hidden1 = nn.Parameter(hidden1)
        self.hidden2 = nn.Parameter(hidden2)
        
        ##LSTM cells for the decoder
        self.lstmcell_prev = nn.LSTMCell(640, 256)
        self.lstmcell_next = nn.LSTMCell(640, 256)
        
        self.wordProject = nn.Linear(in_features=256, out_features=self.num_embeddings)
        
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
    def __init__(self, encoder_model, decoder_model, vocab_size):
        super(UniSKIP_variant, self).__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.vocab_size = vocab_size
        self.embedding_dim = 128
        
        self.wordembed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        
    def forward(self, inputCurr, inputLenghts, inputPrev, inputNext):
        
        # Get the thought of the current sentence
        word_embed_curr = F.tanh(self.wordembed(inputCurr))
        output_ofLastHiddenState = self.encoder(word_embed_curr, inputLenghts)
        
        # Get the embedding for prev and next sentence 
        word_embed_prev = F.tanh(self.wordembed(inputPrev))        
        word_embed_next = F.tanh(self.wordembed(inputNext))
        
        logits_prev, logits_next = self.decoder(word_embed_prev, word_embed_next, output_ofLastHiddenState)
        
        return logits_prev, logits_next

def inference(model, loader, loss):
    i = 0
    epoch_loss = 0
    for batch_idx, (paddedArray, Xlenghts, Yarray, Ylenghts) in enumerate(loader):
        model.hidden = model.hidden_init(paddedArray.size()[1])
        if torch.cuda.is_available():
            paddedArray = paddedArray.cuda()
                    
        X = Variable(paddedArray)
        Y = Yarray.int()
        out = model(X, Xlenghts)
                
        act_lens = torch.from_numpy(np.asarray(Xlenghts)).int()
        label_lens = torch.from_numpy(np.asarray(Ylenghts)).int()
        loss_val = loss(out, Variable(Y+1), Variable(act_lens), Variable(label_lens))
        epoch_loss += loss_val.data[0]
        
        if (i%5 == 0):
            print(loss_val.data[0])
        i = i + 1
            
    print("Validation Loss")
    print(epoch_loss/(i*32))
    return

class Trainer():
    """ A simple training cradle
    """
    
    def __init__(self, model, optimizer, train_loader, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
        #self.loss = CrossEntropyLoss3D()
        self.loss = nn.CrossEntropyLoss().cuda()
        self.loader = train_loader
        
    def stop_cond(self):
        # TODO: Implement early stopping
        def deriv(ns):
            return [ns[i+1] - ns[i] for i in range(len(ns)-1)]
        val_errors = [m.val_error for m in self.metrics]
        back = val_errors[-138:]
        return sum(deriv(back)) > 0
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, n_epochs):
        print("begin training...")
        self.metrics = []
        for e in range(n_epochs):
            if self.stop_cond():
                return
            epoch_loss = 0
            correct = 0
            i = 0
            self.optimizer.zero_grad()
            for batch_idx, (paddedArrayPrev, maskArrayPrev, paddedArrayCurr, Currlenghts, paddedArrayNext, maskArrayNext) in enumerate(self.loader):
                self.model.encoder.hidden = self.model.encoder.hidden_init(paddedArrayCurr.size()[1])
                if torch.cuda.is_available():
                    paddedArrayCurr = paddedArrayCurr.type(torch.LongTensor).cuda()
                    
                Curr_Sen = Variable(paddedArrayCurr)

                #Y = Yarray.int()
                logits_prev, logits_next = self.model(Curr_Sen, Currlenghts, Variable(paddedArrayPrev[:-1,:].type(torch.LongTensor).cuda()), Variable(paddedArrayNext[:-1,:].type(torch.LongTensor).cuda()))

                logits_prev = logits_prev.contiguous().view(-1, logits_prev.size()[2])
                logits_next = logits_next.contiguous().view(-1, logits_next.size()[2])
                
                Y_prev = paddedArrayPrev[1:,:]
                Y_prev = Y_prev.contiguous().view(-1)

                Y_next = paddedArrayNext[1:,:]
                Y_next = Y_next.contiguous().view(-1)

                maskArrayPrev = maskArrayPrev[1:,:]
                maskArrayPrev = maskArrayPrev.contiguous().view(-1)

                maskArrayNext = maskArrayNext[1:,:]
                maskArrayNext = maskArrayNext.contiguous().view(-1)
    
                ind_prev = torch.nonzero(maskArrayPrev, out=None).squeeze()
                ind_next = torch.nonzero(maskArrayNext, out=None).squeeze()
                
                valid_target_prev = torch.index_select(Y_prev, 0, ind_prev).type(torch.LongTensor).cuda()
                valid_output_prev = torch.index_select(logits_prev, 0, Variable(ind_prev.cuda()))

                valid_target_next = torch.index_select(Y_next, 0, ind_next).type(torch.LongTensor).cuda()
                valid_output_next = torch.index_select(logits_next, 0, Variable(ind_next.cuda()))

                loss_prev = self.loss(valid_output_prev, Variable(valid_target_prev))
                loss_next = self.loss(valid_output_next, Variable(valid_target_next))
                
                loss = loss_prev + loss_next
                loss.backward()     
                   
                nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += loss.data[0]
                
                if(i%20 == 0):
                    print(loss.data[0])
                i = i + 1
            
            #if (e%2 == 0):
            #    inference(self.model, self.valLoader, self.loss)
            print("Training Loss")
            print(epoch_loss/(i))
            print("DONE WITH ONE EPOCH")
            print(e)
            wikiTrainer.save_model('./SKIP_THOUGHT_MODEL.pt')
            
            
modelWiki = UniSKIP_variant(9487)

if torch.cuda.is_available(): 
    print("GPU available") 
    modelWiki.cuda()

#modelWiki.load_state_dict(torch.load('./SKIP_THOUGHT_MODEL.pt'))
#optimizer = torch.optim.SGD(modelWiki.parameters(), lr=0.1, momentum=0.9) 
optimizer = torch.optim.Adam(modelWiki.parameters(), lr=0.001, weight_decay=0.00001)
wikiTrainer = Trainer(modelWiki, optimizer, TrainLoader) 
wikiTrainer.run(10)