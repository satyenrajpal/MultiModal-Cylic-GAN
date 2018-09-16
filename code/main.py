from __future__ import print_function
# import torch.backends.cudnn as cudnn
import tensorflow as tf
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
from collections import namedtuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataloader import _use_shared_memory

# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)

import pickle
from miscc.datasets import TextImageDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer
from six.moves import cPickle
import pickle


def collate_fn(data):
    
    ## Sorting based on the utternce lenght
    # middle sentence is the one required to send into the encoder to produce the thoughts
    data.sort(key=lambda x: x[2].shape[0], reverse=True)
    
    ## Unzip the data
    img, X, Y, Z, sentences = zip(*data)
    
    img = [i for i in img]
    img=torch.stack(img,dim=0)
    img=torch.squeeze(img)

    #Sentences are as numpy arrays
    sentences = [i for i in sentences]
    sentences=np.array(sentences)

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

    return img, sentences, paddedArrayX, maskArrayX, paddedArrayY, Ylenghts, paddedArrayZ, maskArrayZ

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='birds_stage1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default=os.getcwd())
    parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101')
    parser.add_argument('--cnn_model_dir', type=str,  default='',
                help='path to resnet file')    
    parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
    parser.add_argument('--cap_dir', type=str,  default='ImgCaptioning',
                help='Path to ImageCaptioning dir')
    parser.add_argument('--vocab_file', type=str, default=None,
        help='manual seed',dest='vocab_file')
    parser.add_argument('--caption_model',type=str,default='topdown',
        dest='caption_model')
    parser.add_argument('--manualSeed', type=int, help='manual seed',dest='manualSeed')
    parser.add_argument('--new_arch',type=int,help='To use new arch or not',dest='new_arch',default=0)
    parser.add_argument('--use_cap_model',type=int, help='To use (1) Captioning Model or not (0)',
        dest='cap_flag',default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()
    if opt.cfg_file is not None:
        cfg_from_file(opt.cfg_file)
    if opt.gpu_id != -1:
        cfg.GPU_ID = opt.gpu_id
    
    # sys.path.append(os.path.abspath(opt.cap_dir))
    # sys.path.append(os.path.abspath(os.path.join(opt.cap_dir,'misc')))

    # import opts as opts
    # import models as models
    # from dataloader import *
    # from dataloaderraw import *
    # import eval_utils as eval_utils
    # import misc.utils as utils
    # import resnet
    # from resnet_utils import myResnet
    # if torch.cuda.is_available():
    #     print("Using CUDA")
    # # print('Using config:')
    # # pprint.pprint(cfg)
    # if opt.manualSeed is None:
    #     opt.manualSeed = random.randint(1, 10000)
    # random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)
    # if cfg.CUDA:
    #     torch.cuda.manual_seed_all(opt.manualSeed)
    # now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # output_dir = '%s/%s_%s_%s' % \
    #              (opt.output_dir,cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    # #mkdir_p(output_dir)     
    # num_gpu = len(cfg.GPU_ID.split(','))
    # vocab_cap=None
    # my_resnet=None    
    # cap_model=None
    # vocab=None
    # #Load image captioning model here

    if opt.vocab_file is not None:
        with open(opt.vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)

    # if opt.cap_flag:
    #     with open(opt.infos_path,'rb') as f:
    #          infos = cPickle.load(f)

        # ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
        # for k in vars(infos[b'opt']).keys():
        #     if k not in ignore:
        #         if k in vars(opt):
        #             assert vars(opt)[k] == vars(infos[b'opt'])[k], k + ' option not consistent'
        #         else:
        #             vars(opt).update({k: vars(infos[b'opt'])[k]}) # copy over options from model
    # with open(opt.infos_path,'rb') as f:
    #      infos = cPickle.load(f,encoding='bytes')

    # ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
    # for k in vars(infos[b'opt']).keys():
    #     if k not in ignore:
    #         if k in vars(opt):
    #             assert vars(opt)[k] == vars(infos[b'opt'])[k], k + ' option not consistent'
    #         else:
    #             vars(opt).update({k: vars(infos[b'opt'])[k]}) # copy over options from model

    # with open(opt.vocab_file, 'rb') as handle:
    #     vocab = cPickle.load(handle)
    # vars(opt).update({'vocab_size':len(vocab)})
    # vars(opt).update({'input_encoding_size':512})#input_encoding_size=512
    # vars(opt).update({'att_feat_size':2048})#=2048, 
    # vars(opt).update({'att_hid_size':512})#att_hid_size=512 , 
    # vars(opt).update({'rnn_size':512})#rnn_size=512
    # vars(opt).update({'rnn_type':'lstm'})#rnn_type='lstm',
    
    # vars(opt).update({'seq_length':16})
    # vars(opt).update({'seq_per_img':5})
    # vars(opt).update({'num_layers':1})
    # vars(opt).update({'drop_prob_lm':0.5})
    # vars(opt).update({'fc_feat_size':2048})
    
#    seq_length=16
 #    seq_per_img=5
    
    # vocab = infos['vocab'] # ix -> word mapping
# (, batch_size=10, beam_size=1, 
#     caption_model='topdown', checkpoint_path='log_td', current_lr=0.00013107200000000006,
#      drop_prob_lm=0.5, fc_feat_size=2048, grad_clip=0.1, id='td', input_att_dir='data/cocotalk_att',
#      9, optim_beta=0.999, optim_epsilon=1e-08,  save_checkpoint_every=3000, scheduled_sampling_increase_every=5, scheduled_sampling_increase_prob=0.05, scheduled_sampling_max_prob=0.25, scheduled_sampling_start=0, self_critical_after=-1, seq_length=,, ss_prob=0.2, 
     # start_from='log_td', train_only=0, val_images_use=5000, vocab_size=9487, weight_decay=0
    # cap_model = models.setup(opt)
    # cap_model.load_state_dict(torch.load(opt.model))
    # cap_model.cuda()
    # cap_model.eval()
    # #ResNet MODEL
    # cnn_model = 'resnet101'
    # my_resnet = getattr(resnet, cnn_model)()
    # my_resnet.load_state_dict(torch.load(opt.cnn_model_dir))
    # my_resnet = myResnet(my_resnet)
    # my_resnet.cuda()
    # my_resnet.eval()
    
    # if opt.vocab_file is None:
    #     print("No vocab file input")
    #     sys.exit()
    # with open(opt.vocab_file, 'rb') as handle:
    #     vocab = cPickle.load(handle)
    
    #     vars(opt).update({'vocab_size':9487})
    #     vars(opt).update({'input_encoding_size':512})#input_encoding_size=512
    #     vars(opt).update({'att_feat_size':2048})#=2048, 
    #     vars(opt).update({'att_hid_size':512})#att_hid_size=512 , 
    #     vars(opt).update({'rnn_size':512})#rnn_size=512
    #     vars(opt).update({'rnn_type':'lstm'})#rnn_type='lstm',
        
    #     vars(opt).update({'seq_length':16})
    #     vars(opt).update({'seq_per_img':5})
    #     vars(opt).update({'num_layers':1})
    #     vars(opt).update({'drop_prob_lm':0.5})
    #     vars(opt).update({'fc_feat_size':2048})
        
    #     seq_length=16
    #     seq_per_img=5
        
    #     vocab_cap = infos['vocab'] # ix -> word mapping
    #     cap_model = models.setup(opt)
    #     cap_model.load_state_dict(torch.load(opt.model))
    #     cap_model.cuda()
    #     cap_model.eval()
    #     #ResNet MODEL
    #     cnn_model = 'resnet101'
    #     my_resnet = getattr(resnet, cnn_model)()
    #     my_resnet.load_state_dict(torch.load(opt.cnn_model_dir))
    #     my_resnet = myResnet(my_resnet)
    #     my_resnet.cuda()
    #     my_resnet.eval()

# (, batch_size=10, beam_size=1, 
#     caption_model='topdown', checkpoint_path='log_td', current_lr=0.00013107200000000006,
#      drop_prob_lm=0.5, fc_feat_size=2048, grad_clip=0.1, id='td', input_att_dir='data/cocotalk_att',
#      9, optim_beta=0.999, optim_epsilon=1e-08,  save_checkpoint_every=3000, scheduled_sampling_increase_every=5, scheduled_sampling_increase_prob=0.05, scheduled_sampling_max_prob=0.25, scheduled_sampling_start=0, self_critical_after=-1, seq_length=,, ss_prob=0.2, 
     # start_from='log_td', train_only=0, val_images_use=5000, vocab_size=9487, weight_decay=0

    if cfg.TRAIN.FLAG:
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([cfg.IMSIZE,cfg.IMSIZE]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # transforms.RandomHorizontalFlip(),
        dataset = TextImageDataset(data_dir=cfg.DATA_DIR,ann_file=cfg.ANN_FILE,
            imsize=cfg.IMSIZE,emb_model=cfg.EMB_MODEL,transform=image_transform,vocab_file=vocab)
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,collate_fn=collate_fn,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(output_dir, cap_model,vocab,eval_utils,my_resnet,
            dataset.word2idx,dataset.emb,dataset.idx2word,vocab_cap=vocab_cap, eval_kwargs=vars(opt))
        algo.train(dataloader, cfg.STAGE)
    else:
        datapath= '%s/test/val_captions.t7' % (cfg.DATA_DIR)
        algo = GANTrainer(output_dir)
        algo.sample(datapath, cfg.STAGE)
