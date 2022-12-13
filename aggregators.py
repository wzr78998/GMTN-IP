import math
import warnings

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import utils
import sklearn
from sklearn.neighbors import KNeighborsClassifier
"""
Set of modules for aggregating embeddings of neighbors.
"""
GPU=1
warnings.filterwarnings("ignore", category=UserWarning)

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features,params, row_list,clum_list,simi_t,simi_s,cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.row=row_list
        self.clum=clum_list
        self.simi_t = simi_t
        self.simi_s = simi_s
        self.use_att=params['use_att']
        self.noise1=params['noise1']
        self.gcn = gcn
        self.domain=None
        self.model=None

        self.label=None
    def forward(self, nodes, to_neighs,domain,model,attention, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        self.domain = domain
        self.model=model
        if domain == 'source':
            row = self.row[0]
            clum = self.clum[0]
            simi = self.simi_s
        if domain == 'target':
            row = self.row[1]
            clum = self.clum[1]
            simi = self.simi_t

        lists_0=[]

        if type(self.features) == type(lists_0):
            if self.domain == 'source':
                features = self.features[0]


            if self.domain == 'target':
                features = self.features[1]



        else:
            features = self.features

        self.label = self.domain
        if not num_sample is None:
            # if num_sample isn't None,random sample num_sample samples in
            # "to_neigh"(the set of neighbors for node in batch)
            # when len(to_neigh)<num_sample,iter all elements in the "to_neigh"
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        #     if num_sample is None:use all information of neigh
            simi_e = []
            if type(nodes) == type(torch.zeros(size=(2, 5))):

                for i in range(nodes.size(0)):
                    node = int(nodes[i])

                    simi_e.append(set(simi[node]))
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [set.union(samp_neigh ,set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        #     得到抽样的neigh中独一的节点，放入Set并编号
        if type(nodes) == type(torch.zeros(size=(2, 5))):
            nodes_lix = torch.zeros(nodes.size(0), 2)
            nodes_lix[:, 0] = torch.tensor(clum[nodes.cpu().detach().numpy()])
            nodes_lix[:, 1] = torch.tensor(clum[nodes.cpu().detach().numpy()])
        unique_nodes_list = list(set.union(*samp_neighs))
        samp_neighs_list1 = []
        for val in unique_nodes_list:
            samp_neighs_list1.append(val)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        d_weight = np.ones(shape=(len(samp_neighs), len(unique_nodes)))
        f_weight = np.ones(shape=(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mask1 = mask.numpy()
        a = max(clum)
        b = max(row)
        if type(nodes)==type(torch.zeros(size=(2,5))):
            for i in range(nodes.size(0)):
                clum_sum=0
                row_sum=0

                ind = np.where(mask1[i] == 1)[0]
                fea_simi=simi_e[i]
                clum_c = clum[int(nodes[i])]/a
                row_c = row[int(nodes[i])]/b
                si_list=list(simi_e[i])
                if len(ind)==len(fea_simi):
                    for j in range(len(ind)):
                        nei = int(samp_neighs_list1[int(ind[j])])
                        imm=np.where((np.floor(np.array(si_list)-nei))==0 )
                        clum_n = clum[nei]/a
                        clum_sum=clum_sum+clum_n
                        row_n = row[nei]/b
                        row_sum = row_sum + row_n
                        if imm[0].shape[0]!=0:
                          f_weight[i, ind[j]]=si_list[imm[0][0]]-nei
                    row_sum=row_sum/len(ind)
                    clum_sum=clum_sum/len(ind)
                    ww=-1/((row_sum-row_c)**2+(clum_sum-clum_c)**2)
                    d_weight[i] = 1 / (1 + np.exp(ww))



        if self.cuda:
            mask = mask.cuda(GPU)
        # 平均操作
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh).cuda(GPU)
        if self.use_att==0:
            # embed_matrix为unique_nodes_list里的各个节点的feature
            if self.noise1==0:
                feature=torch.from_numpy(utils.radiation_noise(features(torch.LongTensor(unique_nodes_list)).numpy())).cuda(GPU)

                feature=torch.tensor(feature,dtype=torch.float)
                embed_matrix = attention(feature)
            else:
                #feature_self = features(torch.LongTensor(nodes)).cuda(GPU)
                feature=features(torch.LongTensor(unique_nodes_list)).cuda(GPU)
               # KNeighborsClassifier.fit(feature_self,np.zeros(shape=(feature.shape[0],1)))

                embed_matrix = (feature)

        else:

            if self.noise1==0:
                feature = torch.from_numpy(utils.radiation_noise(features(torch.LongTensor(unique_nodes_list)).numpy())).cuda(GPU)
                feature = torch.tensor(feature, dtype=torch.float)
                embed_matrix = attention(feature)
            else:
                embed_matrix =( features(torch.LongTensor(unique_nodes_list)).cuda(GPU))
        embed_matrix=torch.tensor(embed_matrix,dtype=torch.float)

        tanh = nn.Tanh().cuda(GPU)
        if type(nodes) == type(torch.zeros(size=(2, 5))):
            mask=mask*torch.tensor( d_weight,dtype=torch.float64).cuda(GPU)*torch.tensor( f_weight,dtype=torch.float64).cuda(GPU)
        to_feats = tanh(torch.tensor(mask,dtype=torch.float32).mm(embed_matrix.cuda(GPU)))
        return to_feats
class MeanAggregator1(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features,params,row_list,clum_list,simi_t,simi_s,cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator1, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.domain=None
        self.model=None
        self.simi_t=simi_t
        self.simi_s = simi_s
        self.row=row_list
        self.clum=clum_list

        self.noise1=params['noise1']

    def forward(self, nodes, to_neighs,domain,model, attention,num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        _set = set
        self.domain=domain
        if domain=='source':
            row=self.row[0]
            clum = self.clum[0]
            simi=self.simi_s
        if domain=='target':
            row=self.row[1]
            clum = self.clum[1]
            simi=self.simi_t
        self.model=model
        if not num_sample is None:
            # if num_sample isn't None,random sample num_sample samples in
            # "to_neigh"(the set of neighbors for node in batch)
            # when len(to_neigh)<num_sample,iter all elements in the "to_neigh"
            _sample = random.sample
            rand=random.sample(range(0,100000),1)[0]
            random.seed(int(rand))
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
            simi_e=[]
            if type(nodes)==type(torch.zeros(size=(2,5))):

                for i in range(nodes.size(0)):


                    node=int(nodes[i])




                    simi_e.append(set(simi[node]))









        #     if num_sample is None:use all information of neigh
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [set.union(samp_neigh ,set([nodes[i]])) for i, samp_neigh in enumerate(samp_neighs)]
        #     得到抽样的neigh中独一的节点，放入Set并编号

        if type(nodes)==type(torch.zeros(size=(2,5))):
            nodes_lix=torch.zeros(nodes.size(0),2)
            nodes_lix[:,0]=torch.tensor(clum[nodes.cpu().detach().numpy()])
            nodes_lix[:, 1] = torch.tensor(clum[nodes.cpu().detach().numpy()])
        unique_nodes_list = list(set.union(*samp_neighs))
        samp_neighs_list1 = []
        for val in unique_nodes_list:
           samp_neighs_list1.append(val)
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        d_weight= np.ones(shape=(len(samp_neighs), len(unique_nodes)))
        f_weight = np.ones(shape=(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        mask1 =mask.numpy()
        a=max(clum)
        b=max(row)
        #c=np.max(fea_simi)
        if type(nodes)==type(torch.zeros(size=(2,5))):
            for i in range(nodes.size(0)):
                clum_sum=0
                row_sum=0

                ind = np.where(mask1[i] == 1)[0]
                fea_simi=simi_e[i]
                clum_c = clum[int(nodes[i])]/a
                row_c = row[int(nodes[i])]/b
                si_list=list(simi_e[i])
                if len(ind)==len(fea_simi):
                    for j in range(len(ind)):
                        nei = int(samp_neighs_list1[int(ind[j])])
                        imm=np.where((np.floor(np.array(si_list)-nei))==0 )
                        clum_n = clum[nei]/a
                        clum_sum=clum_sum+clum_n
                        row_n = row[nei]/b
                        row_sum = row_sum + row_n
                        if imm[0].shape[0]!=0:
                          f_weight[i, ind[j]]=si_list[imm[0][0]]-nei
                    row_sum=row_sum/len(ind)
                    clum_sum=clum_sum/len(ind)
                    ww=-1/((row_sum-row_c)**2+(clum_sum-clum_c)**2)
                    d_weight[i] = 1 / (1 + np.exp(ww))



        if self.cuda:
            mask = mask.cuda(GPU)
        # 平均操作
        num_neigh = mask.sum(1, keepdim=True)
        num_neigh=torch.tensor(num_neigh,dtype=int)
        #替代的
        mask = mask.div(num_neigh).cuda(GPU)
        if False:
            # embed_matrix为unique_nodes_list里的各个节点的feature
            if self.noise1==0:
                embed_matrix = torch.from_numpy(utils.radiation_noise(self.features(torch.LongTensor(nodes)).numpy()))
            else:
                embed_matrix = attention(self.features(torch.LongTensor(nodes)))

        else:

            if self.noise1==0:
                embed_matrix = torch.from_numpy(
                    utils.radiation_noise(self.features(torch.LongTensor(unique_nodes_list),self.domain,self.model).cpu().detach().numpy()))
            else:
                embed_matrix = (self.features(torch.LongTensor(unique_nodes_list),self.domain,self.model))
        embed_matrix=torch.tensor(embed_matrix,dtype=torch.float).cuda(GPU) #维度100
        tanh = nn.Tanh().cuda(GPU)
        if type(nodes) == type(torch.zeros(size=(2, 5))):
            mask=mask*torch.tensor( d_weight,dtype=torch.float64).cuda(GPU)*torch.tensor( f_weight,dtype=torch.float64).cuda(GPU)
        to_feats = tanh(torch.tensor(mask,dtype=torch.float32).mm(embed_matrix.cuda(GPU)))
        return to_feats