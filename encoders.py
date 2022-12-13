import random

import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import utils
GPU=1
import model

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features,attention,feature_dim,
            embed_dim, adj_lists, aggregator,params,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,
            feature_transform=False):
        super(Encoder, self).__init__()
        self.domain=None
        self.use_aug1=params['use_aug1']
        self.features = features
        self.model=None
        self.attention=attention
        self.noise0=params['noise0']
        self.use_att=params['use_att']




        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model
        self.label=None
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)).cuda(GPU)
        init.xavier_uniform_(self.weight)
        # self.weight1 = nn.Parameter(
        #     torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)).cuda(GPU)
        # init.xavier_uniform_(self.weight1)


    def forward(self, nodes,domain,model):
        random_seed = random.sample(range(0, 40), 1)
        random.seed(random_seed[0])
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        self.model=model
        lists0=[]
        self.domain=domain
        if False:
            num_sample = int(np.squeeze(np.random.randint(10,15, (1, 1))))
        else:
           num_sample=self.num_sample

        if type(self.features)==type(lists0):
            if   self.domain=='source':
                features=self.features[0]
                adj_lists=self.adj_lists[0]

            if self.domain=='target':

                features=self.features[1]


                adj_lists = self.adj_lists[1]
        else:
            features=self.features
            adj_lists=self.adj_lists



        self.label = self.domain

        if self.domain == 'target' and self.model == 'train' and self.use_aug1==0:
            nei_list = []
            for node in nodes:

                node_nei = adj_lists[int(node)]
                node_list = []
                for node0 in node_nei:
                    node_list.append(node0)
                rand_int = int(np.squeeze(np.random.randint(0, len(node_list) - 1, (1, 1))))
                node = node_list[rand_int]
                nei_list.append(adj_lists[node])

            neigh_feats = self.aggregator.forward(nodes, nei_list, domain=self.domain,
                                                  model=self.model,attention=self.attention, num_sample=num_sample)

        else:
            neigh_feats = self.aggregator.forward(nodes, [adj_lists[int(node)] for node in nodes], domain=self.domain,
                                                  model=self.model, attention=self.attention,num_sample=num_sample)
        # not gcn - combine self; gcn - only neigh
        if not self.gcn:
            if self.use_att==0:
                if self.noise0==0:
                    self_feats =self.attention(torch.from_numpy(utils.radiation_noise((features(torch.LongTensor(nodes)).numpy()))).cuda(GPU))
                else:
                    self_feats =self.attention(features(torch.LongTensor(nodes)).cuda(GPU))
            else:
                if self.noise0==0:
                    self_feats = torch.from_numpy(utils.radiation_noise(features(torch.LongTensor(nodes)).numpy())).cuda(GPU)
                else:
                    self_feats =(features(torch.LongTensor(nodes)).cuda(GPU))
            self_feats=torch.tensor(self_feats,dtype=torch.float)
            # nom1=nn.BatchNorm1d(100).cuda(GPU)
            # nom2=nn.BatchNorm1d(100).cuda(GPU)
            # self_feats=nom1(self_feats)
            # neigh_feats=nom2(self_feats)

            combined = torch.cat([self_feats.cuda(GPU), neigh_feats.cuda(GPU)], dim=1).cuda(GPU)
        else:
            combined = neigh_feats.cuda(GPU) #维度256
       # attention = F.sigmoid(self.weight1.mm(combined.t()))






        combined = F.tanh(self.weight.mm(combined.t()).t())

        combined=combined+F.tanh(neigh_feats.cuda(GPU))
        return combined.t()

class Encoder1(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features,attention,
            embed_dim, feature_dim, adj_lists, aggregator,params,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,
            feature_transform=False):
        super(Encoder1, self).__init__()
        self.domain=None
        self.attention=attention
        self.use_aug2=params['use_aug2']



        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.label=None
        if base_model != None:
            self.base_model = base_model
        self.model=None
        self.gcn = gcn


        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)
        self.weight1 = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)).cuda(GPU)
        init.xavier_uniform_(self.weight1)

    def forward(self, nodes,domain,model):
        random_seed=random.sample(range(0,40),1)
        random.seed(random_seed[0])
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        self.domain = domain
        self.model=model
        if False:
            num_sample = int(np.squeeze(np.random.randint(self.low, self.high, (1, 1))))
        else:
            num_sample=self.num_sample

        lists_0=[]

        if type(self.adj_lists) == type(lists_0):
            if self.domain == 'source':

                adj_lists = self.adj_lists[0]
            if self.domain == 'target':


                adj_lists = self.adj_lists[1]
        else:

            adj_lists = self.adj_lists
        self.label=self.domain

        #得到样本邻居的feats
        if self.domain == 'target' and self.model == 'train' and self.use_aug2==0:
            nei_list=[]
            for node in nodes:
                node_nei=adj_lists[int(node)]
                node_list=[]
                for node0 in node_nei:
                    node_list.append(node0)
                rand_int = int(np.squeeze(np.random.randint(0, len(node_list)-1, (1, 1))))
                node=node_list[rand_int]
                nei_list.append(adj_lists[node])


            neigh_feats = self.aggregator.forward(nodes, nei_list, domain=self.domain,attention=self.attention,
                                                  model=self.model, num_sample=num_sample)

        else:
            neigh_feats = self.aggregator.forward(nodes, [adj_lists[int(node)] for node in nodes], domain=self.domain,model=self.model,attention=self.attention,num_sample= num_sample)
        # not gcn - combine self; gcn - only neigh
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda(GPU),self.domain,self.model)
            else:
                self_feats = self.features(torch.LongTensor(nodes),self.domain,self.model)




            combined = torch.cat([self_feats, neigh_feats], dim=1).cuda(GPU)
        else:
            combined = neigh_feats #维度200

        combine = F.tanh(self.weight.mm(combined.t()))
        attention=F.sigmoid(self.weight1.mm(combined.t()))
        combine = attention.t()*combine.t()+ attention.t()*F.tanh(neigh_feats.cuda(GPU))
        return combine.t()
