import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        #print("RESNET DETAILS \n\n")
        #print("Input size: ", img.size())
        #x = img.unsqueeze(0)
        x = img
        #print("X size: ", x.size())
        #print("X type: ", type(x))

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #print("before fc layer: ", x.size())
        #fc = x.mean(3).mean(2).squeeze()
        fc = x.mean(3).mean(2)
        #print("after fc layer: ", fc.size())
        att = F.adaptive_avg_pool2d(x,[att_size,att_size])
        #print("Adaptive avg_pool: ", att.size())
        #att = att.squeeze()
        att = att
        #print("After squeezeL ", att.size())
        #att = att.permute(1, 2, 0)
        att = att.permute(0, 2, 3, 1)
        #print("att layer size: ", att.size())
	        
        return fc, att

