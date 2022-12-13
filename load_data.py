import math
from collections import defaultdict

import numpy as np
import os
import pickle
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA


def get_train_test_loader_source(Data_Band_Scaler, GroundTruth):
    print(Data_Band_Scaler.shape)  # (610, 340, 103)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    GroundTruth = GroundTruth.reshape(nRow, -1)
    [Row, Column] = np.nonzero(GroundTruth)
    GroundTruth = np.squeeze(GroundTruth.reshape(1, -1))
    # Sampling samples
    GroundTruth = GroundTruth.reshape(nRow, -1)
    da_train = {}  # Data Augmentation
    m = int(np.max(GroundTruth))  # 9
    index_all = []
    indices1 = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] != 0]
    index_all = index_all + indices1
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] == i + 1]
        if (len(indices) >= 200):
            np.random.shuffle(indices)
            da_train[i] = indices[:200]
    da_train_indices = []
    for i in range(len(da_train)):
        k = i
        da_train_indices += da_train[k]

    da_nTrain = len(da_train_indices)
    imdb = {}
    imdb['data'] = np.zeros([nBand, da_nTrain], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb['set'] = np.zeros([da_nTrain], dtype=np.int64)
    feat_s = {}
    feat_s['data'] = np.zeros([nBand, int(Row.size)], dtype=np.float32)
    feat_s['Labels'] = np.zeros([int(Row.size)], dtype=np.int64)
    RandPerm = da_train_indices
    RandPerm = np.array(RandPerm)
    index_all = np.array(index_all)
    for num in range(int(Row.size)):
        feat_s['data'][:, num] = Data_Band_Scaler[Row[index_all[num]],
                                 Column[index_all[num]], :]
        feat_s['Labels'][num] = GroundTruth[Row[index_all[num]],
                                            Column[index_all[num]]].astype(np.int64)
    for iSample in range(da_nTrain):
        imdb['data'][:, iSample] = Data_Band_Scaler[Row[RandPerm[iSample]],
                                   Column[RandPerm[iSample]], :]
        imdb['Labels'][iSample] = GroundTruth[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)
    imdb['Labels'] = imdb['Labels'] - 1
    feat_s['Labels'] = feat_s['Labels'] - 1
    print('Data is OK.')
    imdb_da_train = imdb
    return feat_s, imdb_da_train, RandPerm, Row, Column,da_train, da_train_indices
def load_source_data(path):
    with open(os.path.join('./datasets', path), 'rb') as handle:
        source_imdb = pickle.load(handle)
    print('chikusei_ok')
    source_data_train = source_imdb[0]  # (2517,2335,128)
    data = source_data_train.reshape(np.prod(source_data_train.shape[:2]), np.prod(source_data_train.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    source_data_train= data_scaler.reshape(source_data_train.shape[0], source_data_train.shape[1], source_data_train.shape[2])
    source_labels_train = source_imdb[1]  # (2517*2335)
    return source_data_train,source_labels_train
def load_data(image_file, label_file,taget_data):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    if taget_data=='IP':
        data_key='Indian_pines'
        label_key = label_file.split('/')[-1].split('.')[0]
    if taget_data=='UP':
        data_key = 'paviaU'
        label_key = 'paviaU_gt'
    if taget_data=='SA':
        data_key = 'salinas_corrected'
        label_key = 'salinas_gt'
    if taget_data=='PC':
        data_key = 'pavia'
        label_key = 'pavia_gt'
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth

def adj_list(spatial_similarity):
    m, n = spatial_similarity.shape
    adj_lists = defaultdict(set)
    for i in range(m):
        for ind in np.where(spatial_similarity[i] == 1)[0]:
            adj_lists[i].add(ind)
    return adj_lists

def applyPCAs(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[1])) ##沿光谱维度展平
    pca = PCA(n_components=numComponents, whiten=False)
    newX = pca.fit_transform(newX.T)
    newX = np.reshape(newX, (X.shape[1],numComponents))
    return newX, pca
def get_train_data(imdb_da_train_s,feat_s,PCA_dim):
    imdb_source_data = imdb_da_train_s['data']  # 源域的元数据
    imdb_source_data,_=applyPCAs(imdb_source_data,PCA_dim)
    imdb_source_label = imdb_da_train_s['Labels']  # 源域的元数据标签
    del imdb_da_train_s
    adj=np.load('./datasets/spatial_simislarity_source.npy')
    adj=np.array(adj,dtype=np.int8)

    adj_lists=adj_list(adj)
    del adj
    feat_s['data'], _ = applyPCAs(feat_s['data'], PCA_dim)
    feat_data_s = feat_s['data']  #
    feat_label_s = feat_s['Labels']
    del feat_s

    return  imdb_source_data,imdb_source_label,feat_data_s,feat_label_s ,adj_lists
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) # (610, 340, 103)

    [nRow, nColumn, nBand] = Data_Band_Scaler.shape
    [Row, Column] = np.nonzero(GroundTruth)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(GroundTruth))  # 9
    nlabeled =shot_num_per_class

    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)
    index_all=[]

    indices1 = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] != 0]
    index_all=index_all+indices1
    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if GroundTruth[Row[j], Column[j]] == i + 1]

        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))  # 520
    print('the number of test_indices:', len(test_indices))  # 9729
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([ nBand, nTrain + nTest], dtype=np.float32)  # (9,9,100,n)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices


    RandPerm = np.array(RandPerm)
    index_all ==np.array(index_all)

    for iSample in range(nTrain + nTest):
        imdb['data'][ :, iSample] = Data_Band_Scaler[Row[index_all[iSample]],
                                         Column[index_all[iSample]], :]
        imdb['Labels'][iSample] = GroundTruth[Row[index_all[iSample]], Column[index_all[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1  # 1-16 0-15
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')




    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([da_nTrain,nBand ],  dtype=np.float32)  # (9,9,100,n)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  # radiation_noise，flip_augmentation
        imdb_da_train['data'][ iSample, ] = Data_Band_Scaler[Row[da_RandPerm[iSample]],
            Column[da_RandPerm[iSample]] , :]
        imdb_da_train['Labels'][iSample] = GroundTruth[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  # 1-16 0-15
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return imdb, imdb_da_train ,RandPerm,Row, Column,nTrain,train,test,da_train,train_indices,test_indices,da_train_indices


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    np.random.seed(1236)
    imdb, imdb_da_train,RandPerm,Row, Column,nTrain ,train,test,da_train,train_indices,test_indices,da_train_indices = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class


    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation


    target_da_datas = imdb_da_train['data']  # (9,9,100, 1800)->(1800, 100, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  # (1800,)
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification


    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)  # (9, 9, 100, 225)
    print(imdb_da_train['Labels'])


    return imdb,RandPerm,Row, Column,nTrain,train,test,da_train,train_indices,test_indices,da_train_indices
def get_target_data(taget_data):
    if taget_data=='IP':
        test_data = './datasets/IP/Indian_pines_corrected.mat'
        test_label = './datasets/IP/indian_pines_gt.mat'
        adj=np.load('./datasets/spatial_simislarity_IP.npy')
        adj_lists=adj_list(adj)
    del adj
    Data_Band_Scaler, GroundTruth=load_data(test_data,test_label,taget_data)
    return  Data_Band_Scaler,GroundTruth,adj_lists
