from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import numpy as np
import torchfile,random

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from tensorboard import summary
#from tensorboard import FileWriter
from ConcurrentThoughtsModel import *
from logger import *

# import eval_utils as eval_utils

class GANTrainer(object):
    def __init__(self, output_dir,cap_model,vocab,eval_utils,my_resnet,new_arch,eval_kwargs={}):#
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            #self.summary_writer = FileWriter(self.log_dir)
        self.cap_model=cap_model
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = 5
        self.gpus=None
        if cfg.GPU_ID is not None:
            s_gpus = cfg.GPU_ID.split(',')
            self.gpus = [int(ix) for ix in s_gpus]

        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.vocab=vocab
        self.eval_kwargs=eval_kwargs
        self.eval_utils=eval_utils
        self.my_resnet=my_resnet
        ##layers, hidden size, bidirectional, emb_dim
        self.CTencoder= STEncoder(1, 512, True, 128)

        ##hidden_size, embedding_dim, thought_size, vocab_size
        self.CTdecoder = STDuoDecoderAttn(256, 128, 512, 9487)

        ##encoder_model, decoder_model, embedding_dim, vocab_size
        self.CTallmodel = UniSKIP_variant(self.CTencoder, self.CTdecoder, 128, 9487)

        self.cosEmbLoss = nn.CosineEmbeddingLoss()
        ## Loss Function
        self.CTloss = nn.CrossEntropyLoss()
        if(cfg.CUDA):
            self.cosEmbLoss=self.cosEmbLoss.cuda()
            self.CTloss = self.CTloss.cuda()
            self.CTencoder = self.CTencoder.cuda()
            self.CTdecoder = self.CTdecoder.cuda()
            self.CTallmodel = self.CTallmodel.cuda()
        self.new_arch=new_arch

    # ############# For training stageI GAN ############# No Need to do anything
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        print(netG)
        netD = STAGE1_D(self.new_arch)
        netD.apply(weights_init)
        print(netD)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CTModel !='':
            state_dict=\
                torch.load(cfg.CTModel,
                    map_location=lambda storage, loc: storage)
            self.CTallmodel.load_state_dict(state_dict)
            print('Load CT Model From: ',cfg.CTModel)
        print(self.CTallmodel)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  ############# No need to do anything
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        print(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        print(netD)
        if cfg.CTModel !='':
            state_dict=\
                torch.load(cfg.CTModel,
                    map_location=lambda storage, loc: storage)
            self.CTallmodel.load_state_dict(state_dict)
            print('Load CT Model From: ',cfg.CTModel)
        print(self.CTallmodel)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD


    def train(self, data_loader, stage=1):
        logger = Logger('./logs_CS_GAN')       
        image_transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([64, 64]),
            transforms.ToTensor()])
        CT_update=35 if cfg.CTModel== '' else 0 
        print("Training CT model for ",CT_update)
        
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()
        
        #######
        nz = cfg.Z_DIM
        batch_size = self.batch_size
        # flags = Variable(torch.cuda.FloatTensor([-1.0]*batch_size))
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),
                     volatile=True)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        #Gaussian noise input added to the input images to the disc
        noise_input= Variable(torch.zeros(batch_size,3,cfg.FAKEIMSIZE,cfg.FAKEIMSIZE))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()
            noise_input=noise_input.cuda()
            # flags.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.SGD(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR)
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        ####Optimizers for CT c                             ##########################TODO:PRINT PARAMETERS!!!!
        optimizerCTallmodel=optim.Adam(self.CTallmodel.parameters(),
                                lr=0.0001, weight_decay=0.00001,betas=(0.5, 0.999))
        optimizerCTenc=optim.Adam(self.CTencoder.parameters(),
                                lr=0.0001, weight_decay=0.00001,betas=(0.5, 0.999))
        count = 0
        len_dataset=len(data_loader)
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
            print("Started training for new epoch")
            optimizerCTallmodel.zero_grad()
            ct_epoch_loss=0
            emb_loss=0
            epoch_count=0
            for i, data in enumerate(data_loader):
                ######################################################
                # (1) Prepare training data
                ######################################################

                real_img_cpu, sentences, paddedArrayPrev, maskArrayPrev, paddedArrayCurr, Currlenghts, paddedArrayNext, maskArrayNext = data
                self.CTallmodel.encoder.hidden=self.CTallmodel.encoder.hidden_init(paddedArrayCurr.size(1))
                real_imgs = Variable(real_img_cpu)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    paddedArrayCurr=Variable(paddedArrayCurr.type(torch.LongTensor).cuda())
                    paddedArrayNext_input=Variable(paddedArrayNext[:-1, :].type(torch.LongTensor).cuda())
                    paddedArrayPrev_input=Variable(paddedArrayPrev[:-1, :].type(torch.LongTensor).cuda())
                
                sent_hidden, logits_prev, logits_next = self.CTallmodel(paddedArrayCurr, Currlenghts, paddedArrayPrev_input, paddedArrayNext_input)

                #Optimizing over Concurrent model
                if (epoch < CT_update) :
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
                    
                    valid_target_prev = torch.index_select(Y_prev, 0, ind_prev).type(torch.LongTensor)
                    valid_output_prev = torch.index_select(logits_prev, 0, Variable(ind_prev).cuda())

                    valid_target_next = torch.index_select(Y_next, 0, ind_next).type(torch.LongTensor)
                    valid_output_next = torch.index_select(logits_next, 0, Variable(ind_next).cuda())

                    loss_prev = self.CTloss(valid_output_prev, Variable(valid_target_prev).cuda())
                    loss_next = self.CTloss(valid_output_next, Variable(valid_target_next).cuda())
                    
                    self.CTallmodel.zero_grad()
                    optimizerCTallmodel.zero_grad()
                    loss = loss_prev + loss_next
                    loss.backward(retain_graph=True)
                    ct_epoch_loss += loss.data[0]
                    nn.utils.clip_grad_norm(self.CTallmodel.parameters(), 0.25)
                    optimizerCTallmodel.step()
         
                if epoch>=CT_update:
                #######################################################
                # (2) Generate fake images
                ######################################################
                    noise.data.normal_(0, 1)
                    inputs = (sent_hidden, noise)
                    
                    _, fake_imgs, mu, logvar = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus) #### TODO: Check Shapes!!!!->Checked
                    
                    # _,fake_imgs,mu,logvar=netG(inputs[0],inputs[1])
                    #######################################################
                    # (2.1) Generate captions for fake images
                    ######################################################
                    # sents,h_sent=self.eval_utils.captioning_model(real_imgs,self.cap_model,self.vocab,self.my_resnet,self.eval_kwargs)
                    # h_sent_var=Variable(torch.FloatTensor(h_sent)).cuda()
                    # input_layer = tf.stack([preprocess_for_train(i) for i in real_imgs], axis=0)
                    real_imgs = Variable(torch.stack([image_transform_train(img.data.cpu()).cuda() for img in real_imgs], dim=0))

                    ############################
                    # (3) Update D network
                    ###########################
                    """
                    if np.isnan(loss_cos.data.cpu().numpy()[0]):
                        print("\n\n")
                        print("Embedding Loss: ", loss_cos.data[0])
                        print("\n")
                        print("Sent Hidden min: ", torch.min(sent_hidden))
                        print("H_Sent: ", torch.min(h_sent_var))
                        print("Sentences Input min: ", sentences)
                        print("VCS Input: ", sents)
                        print("\n\n")
                    """
                    if random.uniform(0,1)<0.9:
                        noise_input.data.normal_(0,1)
                        fake_imgs=fake_imgs+noise_input
                        real_imgs=real_imgs+noise_input
                    # if epoch>20 :
                        # self.CTallmodel.zero_grad()
                    netD.zero_grad()
                    errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                               real_labels, fake_labels,
                                               mu, self.gpus)
                    errD.backward()
                    optimizerD.step()

                    #Label Switching
                    #Trick as of - https://github.com/soumith/ganhacks/issues/14
                    netD.zero_grad()
                    errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                               fake_labels, real_labels,
                                               mu, self.gpus)
                    errD.backward()
                    optimizerD.step()
                    ############################
                    # (4) Update G network
                    ###########################
                    # loss_cos=self.cosEmbLoss(sent_hidden, h_sent_var,flags)
                    netG.zero_grad()
                    errG = compute_generator_loss(netD, fake_imgs,
                                                  real_labels, mu, self.gpus)
                    kl_loss = KL_loss(mu, logvar)
                    errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                    # else :
                    #     errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL + 10*loss_cos
                    errG_total.backward()
                    optimizerG.step()
                    # emb_loss += loss_cos.data[0]

                count = count + 1
                epoch_count+=1
                if i % 100 == 0:
                    print("Loss CT Model: ", ct_epoch_loss/epoch_count)
                    # print("Emb Loss: ", emb_loss)
            # save the image result for each epoch after embedding model has been trained
            if epoch>=CT_update:                
                inputs = (sent_hidden, fixed_noise)

                lr_fake, fake, _, _ = \
                            nn.parallel.data_parallel(netG, inputs, self.gpus)
                save_img_results(real_img_cpu, fake, epoch, self.image_dir,sentences)
                if lr_fake is not None:
                    save_img_results(None, lr_fake, epoch, self.image_dir,sentences)
            
                end_t = time.time()
                
                print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.data[0], errG.data[0], kl_loss.data[0],
                     errD_real, errD_wrong, errD_fake, (end_t - start_t)))
                # logger.scalar_summary('Cosine_loss', emb_loss, epoch+1)
                logger.scalar_summary('errD_loss', errD.data[0]/len_dataset, epoch+1)
                logger.scalar_summary('errG_loss', errG.data[0]/len_dataset, epoch+1)
                logger.scalar_summary('kl_loss', kl_loss.data[0]/len_dataset, epoch+1)

            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, self.CTallmodel, epoch, self.model_dir)
            logger.scalar_summary('CT_loss', ct_epoch_loss/len_dataset, epoch+1)

        save_model(netG, netD, self.CTallmodel, self.max_epoch, self.model_dir)

    def sample(self, datapath, stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            t
            _, fake_imgs, mu, logvar = \
                nn.parallel.data_parallel(netG, inputs, self.gpus)
            for i in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

