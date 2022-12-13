import torch
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def get_samples(train_data_1,feat_data2,max_class,CLASS_NUM,SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS,domain_batch):
    data1_class_choose = random.sample(range(0,max_class),CLASS_NUM)
    support_sample = np.zeros(shape=(CLASS_NUM, SHOT_NUM_PER_CLASS))
    query_sample = np.zeros(shape=(CLASS_NUM, QUERY_NUM_PER_CLASS))
    support_labels= np.zeros(shape=support_sample.shape)
    query_labels = np.zeros(shape=query_sample.shape)
    data2_sample=np.random.randint(0,feat_data2.shape[0],size=(domain_batch,))
    for i in range(CLASS_NUM):
        class_choose = data1_class_choose[i]
        random_sample_choose= random.sample(range(0, 200), SHOT_NUM_PER_CLASS+QUERY_NUM_PER_CLASS)
        random_sample_choose_suport = random_sample_choose[:SHOT_NUM_PER_CLASS]
        random_sample_choose_query = random_sample_choose[SHOT_NUM_PER_CLASS:]

        support_sample[i] = train_data_1[[class_choose], [random_sample_choose_suport]]
        support_labels[i] = i
        query_sample[i] = train_data_1[[class_choose], [random_sample_choose_query]]
        query_labels[i] = i

    support_sample= torch.LongTensor(np.squeeze(support_sample.reshape(1, -1)))
    query_sample = torch.LongTensor(np.squeeze(query_sample.reshape(1, -1)))
    data2_sample = torch.LongTensor(np.squeeze(data2_sample.reshape(1, -1))).long()
    query_labels = np.squeeze(query_labels.reshape(1, -1))
    support_labels= np.squeeze(support_labels.reshape(1, -1))
    # 打乱测试集样本顺序
    permution = np.random.permutation(np.arange(CLASS_NUM* 19))
    query_sample= query_sample[permution]
    query_labels = query_labels[permution]
    return support_sample,query_sample,support_labels,query_labels,data2_sample

