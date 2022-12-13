import math

import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.sparse
GPU=1
class Attention1_nei(nn.Module):
    def __init__(self, in_dimension,squeeze=4):
        super(Attention1_nei, self).__init__()
        self.ave=nn.AdaptiveAvgPool1d(1)
        self.preconv1 = nn.Linear(in_dimension,in_dimension//squeeze)
        self.preconv2 = nn.Linear(in_dimension//squeeze,in_dimension)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        out=x.unsqueeze(2)
        out=out.permute(2,1,0)
        out=self.ave(out)
        out=out.squeeze(0)
        out=out.permute(1,0)

        out= self.preconv1(out)
        out=self.relu(out)
        out=self.preconv2(out)
        out=self.sigmoid(out)


        return out*x
class Attention_center(nn.Module):
    def __init__(self, in_dimension):
        super(Attention_center, self).__init__()

        self.preconv1 = nn.Linear(in_dimension*2,in_dimension)
        self.preconv2 = nn.Linear(in_dimension,in_dimension)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Tanh()
    def forward(self, x,nei):
        # out=nei.unsqueeze(2)
        # out=out.permute(2,1,0)
        # out=self.ave(out)
        # out=out.squeeze(0)
        # out=out.permute(1,0)

        out= self.preconv1(torch.cat([x,nei],1))
        out=self.relu(out)
        out=self.preconv2(out)
        out=self.sigmoid(out)
        return out*x,out*nei
class prot(nn.Module):
    def __init__(self, in_dimension,out_dim):
        super(prot, self).__init__()

        self.preconv1 = nn.Linear(in_dimension,in_dimension)
        self.preconv2 = nn.Linear(in_dimension,out_dim)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()

    def forward(self, x):


        x= self.preconv1(x)
        x=self.relu(x)
        x=self.preconv2(x)
        x=self.tanh(x)


        return x
class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.domain=None
        self.model=None
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)



    def forward(self, nodes,domain,model):

        self.domain=domain
        self.model=model
        embeds = self.enc(nodes,self.domain,self.model)
        scores = self.weight.mm(embeds)

        return embeds.t(),scores.t()
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1
class DomainClassifier(nn.Module):
    def __init__(self,domain_dim,dropout):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(domain_dim, domain_dim), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(domain_dim, domain_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(domain_dim, domain_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(domain_dim, domain_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

        )
        self.domain = nn.Linear(domain_dim, 1) # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
        x.register_hook(grl_hook(coeff)) #梯度反转层
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):

        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]

        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self,GPU):
        super(RandomLayer, self).cuda(GPU)
        self.random_matrix = [val.cuda(GPU) for val in self.random_matrix]

