import argparse
import random
import time
import warnings
import numpy as np
import torch
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
import aggregators
import encoders
import utils
import os
import load_data
import model




warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--target_data', type=str, default='IP',
                    help='dataset')
parser.add_argument('--path', type=str, default='Chikusei_imdb_128.pickle',
                    help='path')
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 220) 
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 20000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
# target
parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')
args = parser.parse_args(args=[])
def get_params():
    # Training settings
    parser = argparse.ArgumentParser()  # argparse命令行选项、参数和子命令解析器，后面是对象
    parser.add_argument('--enc1_size', type=int, default=100, metavar='N',
                        help='embed_dim1')
    parser.add_argument('--enc2_size', type=int, default=100, metavar='N', help='embed_dim2')  
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_samples1', type=int, default=30,
                        help='num_samples1.')  # 5e-18
    parser.add_argument('--num_samples2', type=int, default=30,
                        help='num_samples2.')
    parser.add_argument('--low', type=int, default=0,
                        help='low.')  # 5e-18
    parser.add_argument('--high', type=int, default=0,
                        help='high.')
    parser.add_argument('--dim', type=int, default=100,
                        help='dim.')
    parser.add_argument('--features_dim', type=int, default=100,
                        help='features_dim.')
    parser.add_argument('--train_num', type=int, default=400,
                        help='train_num.')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                        help='weight_decay.')
    parser.add_argument('--squeeze', type=int, default=1,
                        help='squeeze.')
    parser.add_argument('--use_aug2', type=int, default=0,
                        help='squeeze.')
    parser.add_argument('--use_aug1', type=int, default=0,
                        help='squeeze.')
    parser.add_argument('--noise0', type=int, default=1,
                        help='noise0.')
    parser.add_argument('--noise1', type=int, default=1,
                        help='noise1.')
    parser.add_argument('--use_att', type=int, default=0,
                        help='attention.')
    parser.add_argument('--domain_dim', type=int, default=128,
                        help='domain_dim.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout.')
    parser.add_argument('--domain_batch', type=int, default=128,
                        help='dropout.')
    args, _  = parser.parse_known_args()
    return args
last_accuracy = 0.0
best_episdoe = 0
strong_acc=0.0
train_loss = []
test_acc = []
running_D_loss, running_F_loss = 0.0, 0.0
running_label_loss = 0
running_domain_loss = 0
total_hit, total_num = 0.0, 0.0
test_acc_list = []
params = vars(get_params())


SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
# Hyper Parameters in target domain data set
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1
utils.same_seeds(0)
def _init_():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())
def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
#获得源域的数据
source_data_train,source_labels_train=load_data.load_source_data(args.path)

feat_s,imdb_da_train_s, RandPerm_s,Row_s, Columns_s,da_train_s,da_train_indices_s=load_data.get_train_test_loader_source(source_data_train,source_labels_train)
imdb_source_data, imdb_source_label, feat_data_s, feat_label_s, adj_lists_s=load_data.get_train_data(imdb_da_train_s,feat_s,PCA_dim=params['dim'])
simi_list_s= np.load('./simi_list_s.npy',allow_pickle=True) .tolist()


del feat_s
del source_data_train
del source_labels_train

#获得目标域数据
Data_Band_Scaler, GroundTruth ,adj_lists_t= load_data.get_target_data(args.target_data)
nDataSet = 1
simi_list_t= np.load('./simi_list_t.npy',allow_pickle=True)  .tolist()


acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
kx = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
for iDataSet in range(nDataSet):
    # 划分目标域数据
    imdb_target, RandPerm,Row, Column,nTrain,train,test,da_train,train_indices,test_indices,da_train_indices = load_data.get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    del Data_Band_Scaler
    feat_data_t = imdb_target['data']
    feat_data_t, _ = load_data.applyPCAs(feat_data_t, params['dim'])
    feat_label_t = imdb_target['Labels'].T
    del imdb_target
    # 目标域的带标签数据：
    [row_s, column_s] = [Row_s, Columns_s]
    row_list = [row_s, Row]                          
    clum_list = [column_s, Column]

    train_data = train_indices  # 训练集索引
    test_data = test_indices  # 测试集索引
    # 目标域的元训练数据`
    train_da = da_train  # 按类存放的元训练数据索引
    train_da_t = np.zeros((CLASS_NUM, 200))
    for key, value in train_da.items():
        train_da_t[key, :] = value



    [row_s, column_s] = [Row_s, Columns_s]
    row_list = [row_s, Row]
    clum_list = [column_s, Column]
    da_train_class = da_train_s  # 元数据序列按类存放的字典
    train_da_s = np.zeros((18, 200))
    for key, value in da_train_class.items():
        train_da_s[key, :] = value

    features_s = nn.Embedding(feat_data_s.shape[0], params['dim'])
    features_t = nn.Embedding(feat_data_t.shape[0],params['dim'])
    features_s.weight = nn.Parameter(torch.FloatTensor(feat_data_s), requires_grad=False)
    features_t.weight = nn.Parameter(torch.FloatTensor(feat_data_t), requires_grad=False)
    num_samples1 = params['num_samples1']
    num_samples2 = params['num_samples2']
    enc1_size = params['enc1_size']
    enc2_size = params['enc2_size']
    adj_lists_lists = [adj_lists_s, adj_lists_t]
    del adj_lists_s
    del adj_lists_t
    features_list = [features_s, features_t]
    del features_s
    del features_t
    attention = model.Attention_center(params['dim'])
    attention1 = model.Attention1_nei(params['dim'])
    agg1 = aggregators.MeanAggregator(features_list, params, row_list,clum_list,simi_list_t, simi_list_s,   cuda=False)
    enc1 = encoders.Encoder(features_list, attention1, params['dim'], enc1_size, adj_lists_lists, agg1, params,
                                      gcn=False, cuda=False, num_sample=num_samples1)
    agg2 = aggregators.MeanAggregator1(lambda nodes, domain, model: enc1(nodes, domain, model).t(), params,row_list,clum_list,simi_list_t, simi_list_s,
                                                   cuda=False)




    enc2 = encoders.Encoder1(lambda nodes, domain, model: enc1(nodes, domain, model).t(), attention,
                                       enc2_size, enc1.embed_dim, adj_lists_lists, agg2, params,
                                       base_model=enc1, gcn=False, cuda=False, num_sample=num_samples2)
    del adj_lists_lists
   

    graphsage = model.SupervisedGraphSage(TEST_CLASS_NUM, enc2)
    domain_classifier = model.DomainClassifier(params['domain_dim'],dropout=params['dropout'])
    domain_classifier.apply(weights_init)
    random_layer = model.RandomLayer([params['enc2_size'], args.class_num], params['domain_dim'])
    random_layer.cuda(GPU)
    graphsage.cuda(GPU)
    domain_classifier.cuda(GPU)
    domain_classifier_optim = torch.optim.RMSprop(domain_classifier.parameters(), lr=params['lr'],
                                                  weight_decay=params['weight_decay'])

    optimizer_g = torch.optim.RMSprop(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=params['lr'],
                                      weight_decay=params['weight_decay'])
    crossEntropy = nn.CrossEntropyLoss().cuda(GPU)

    domain_criterion = nn.BCEWithLogitsLoss().cuda(GPU)
    print("Training...")


    for episode in range(params['train_num']) :
        if episode%2==0:
            '''Few-shot claification for source domain data set'''




            support_sample_s,query_sample_s,support_labels_s,query_labels_s,target_sample=utils.get_samples(train_da_s,feat_data_t,18,CLASS_NUM,SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS,domain_batch=params['domain_batch'])
            target_features,target_outputs = graphsage.forward(target_sample, 'target','train')

            support_features_s,support_outputs_s=graphsage.forward(support_sample_s,'source','train')

            query_features_s,query_outputs_s=graphsage.forward(query_sample_s,'source','train')

            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto_s = support_features_s.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto_s = support_features_s

            logits = euclidean_metric(query_features_s, support_proto_s) #查询集特征图和原型之间的 距离
            query_labels_s=torch.tensor(query_labels_s,dtype=torch.long)
            f_loss = crossEntropy(logits,query_labels_s.cuda(GPU))
            '''domain adaptation'''

            features = torch.cat([support_features_s, query_features_s, target_features], dim=0) #把支持集查询集还有目标域特征图拼接在一起 (16+16*19+128,160)
            outputs = torch.cat((support_outputs_s, query_outputs_s, target_outputs), dim=0) #把支持集查询集还有目标域输出拼接在一起 (16+16*19+128,16)
            softmax_output = nn.Softmax(dim=1)(outputs)



            domain_label = torch.zeros([support_sample_s.shape[0] + query_sample_s.shape[0] + target_sample.shape[0], 1]).cuda(GPU)
            domain_label[:support_sample_s.shape[0] + query_sample_s.shape[0]] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)
            loss = f_loss +domain_loss
            graphsage.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            optimizer_g.step()

            domain_classifier_optim.step()
            label='target'

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels_s).item()
            total_num += query_sample_s.shape[0]



        else:

            support_sample_t, query_sample_t, support_labels_t, query_labels_t, source_sample = utils.get_samples(
                train_da_t, feat_data_s, 16, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS,
                domain_batch=params['domain_batch'])
            source_features, source_outputs = graphsage.forward(source_sample, 'source', 'train')
            support_features_t, support_outputs_t = graphsage.forward(support_sample_t, 'target', 'train')
            query_features_t, query_outputs_t = graphsage.forward(query_sample_t, 'target', 'train')
            # Prototype network
            if SHOT_NUM_PER_CLASS > 1:
                support_proto_t= support_features_t.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)  # (9, 160)
            else:
                support_proto_t = support_features_t

            logits = euclidean_metric(query_features_t, support_proto_t)  # 查询集特征图和原型之间的 距离
            query_labels_t= torch.tensor(query_labels_t, dtype=torch.long)
            f_loss = crossEntropy(logits, query_labels_t.cuda(GPU))

            features = torch.cat([support_features_t, query_features_t, source_features],
                                 dim=0)
            outputs = torch.cat((support_outputs_t, query_outputs_t, source_outputs),
                                dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)


            domain_label = torch.zeros(
                [support_sample_t.shape[0] + query_sample_t.shape[0] + source_sample.shape[0], 1]).cuda(GPU)
            domain_label[support_sample_t.shape[0] + query_sample_t.shape[0]:] = 1

            randomlayer_out = random_layer.forward(
                [features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)

            loss = f_loss + domain_loss

            graphsage.zero_grad()

            domain_classifier.zero_grad()
            loss.backward()
            optimizer_g.step()


            domain_classifier_optim.step()
            label = 'source'

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels_t).item()
            total_num += query_sample_t.shape[0]



        if (episode + 1) % 1 == 0:
            print('episode:',episode)
        if (episode + 1) % 100 == 0 :
            print("Testing ...")
           

            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            train = np.zeros(shape=(CLASS_NUM, TEST_LSAMPLE_NUM_PER_CLASS))
            for k in range(CLASS_NUM):
                train[k] = train_data[k * TEST_LSAMPLE_NUM_PER_CLASS:(k + 1) * TEST_LSAMPLE_NUM_PER_CLASS]
            train_sample = train
            train_sample = torch.LongTensor(np.squeeze(train_sample.reshape(1, -1)))
            train_labels = feat_label_t[train_sample]
            train_features ,_= graphsage(train_sample, 'target', 'test')
            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)  # 对特征图归一化
            KNN_classifier = KNeighborsClassifier(n_neighbors=1, p=2)
            KNN_classifier.fit(train_features.cpu().detach().numpy(),
                               np.squeeze(train_labels.reshape(1, -1)))  
            # 测试集有100个数据
            batch_size = 100
            features_all=np.zeros(shape=(len(test_data),train_features.cpu().detach().numpy().shape[1]))
            for num in range(len(test_data) // batch_size+1):
                if (num == len(test_data) // batch_size):
                    test_sampel = test_data[num * batch_size:]
                test_sampel = test_data[num * batch_size:(num + 1) * batch_size]
                test_labels = feat_label_t[test_sampel]
                test_features ,_= graphsage(test_sampel, 'target', 'test')
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(len(test_labels))]
                features_all[counter:counter+len(test_labels),:]=test_features.cpu().detach().numpy()
                total_rewards += np.sum(rewards)
                counter += len(test_labels)
                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)
                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / (len(test_data))
            print("best_acc:", last_accuracy)
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
            acc[iDataSet] = 100. * total_rewards / (len(test_data))
            OA = acc
            AA = np.mean(A, 1)
            kx[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print("acc:", test_accuracy)
            print("A", A)
            print("AA:", AA)
            print("kappa:", kx)
            if test_accuracy > last_accuracy:
                last_accuracy=test_accuracy
                print("save networks for episode:", episode)
                torch.save(graphsage.state_dict(),str('./checkpoints/'+'IP_acc:_'+str(round(last_accuracy,2))+'_.pkl'))
                print('acc:', test_accuracy)






