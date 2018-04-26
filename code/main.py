from __future__ import print_function
import torch.backends.cudnn as cudnn
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


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextImageDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer
from six.moves import cPickle

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
    parser.add_argument('--manualSeed', type=int, help='manual seed',dest='manualSeed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = parse_args()
    if opt.cfg_file is not None:
        cfg_from_file(opt.cfg_file)
    if opt.gpu_id != -1:
        cfg.GPU_ID = opt.gpu_id
    
    sys.path.append(os.path.abspath(opt.cap_dir))
    sys.path.append(os.path.abspath(os.path.join(opt.cap_dir,'misc')))

    import opts as opts
    import models as models
    from dataloader import *
    from dataloaderraw import *
    import eval_utils as eval_utils
    import misc.utils as utils
    import resnet
    from resnet_utils import myResnet

    #if args.data_dir != '':
     #   cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(opt.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '%s/%s_%s_%s' % \
                 (opt.output_dir,cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    #mkdir_p(output_dir)     
    num_gpu = len(cfg.GPU_ID.split(','))

    #Load image captioning model here
    with open(opt.infos_path) as f:
        infos = cPickle.load(f)
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model


    vocab = infos['vocab'] # ix -> word mapping

    cap_model = models.setup(opt)
    cap_model.load_state_dict(torch.load(opt.model))
    cap_model.cuda()
    cap_model.eval()
    #ResNet MODEL
    cnn_model = opt.cnn_model
    my_resnet = getattr(resnet, cnn_model)()
    my_resnet.load_state_dict(torch.load('./ImgCaptioning/data/imagenet_weights/'+cnn_model+'.pth'))
    my_resnet = myResnet(my_resnet)
    my_resnet.cuda()
    my_resnet.eval()

    if cfg.TRAIN.FLAG:
        image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([cfg.IMSIZE,cfg.IMSIZE]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = TextImageDataset(data_dir=cfg.DATA_DIR,ann_file=cfg.ANN_FILE,
                              imsize=cfg.IMSIZE,emb_model=cfg.EMB_MODEL,
                              transform=image_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(output_dir, cap_model,vocab,eval_utils,my_resnet,eval_kwargs=vars(opt))
        algo.train(dataloader, cfg.STAGE)
    else:
        datapath= '%s/test/val_captions.t7' % (cfg.DATA_DIR)
        algo = GANTrainer(output_dir)
        algo.sample(datapath, cfg.STAGE)
