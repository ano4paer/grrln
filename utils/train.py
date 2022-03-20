import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from sklearn.metrics import classification_report
from visdom import Visdom
from sklearn.utils import shuffle as reset

warnings.filterwarnings('ignore')


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx * bs: (idx + 1) * bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, torch.FloatTensor(labels)


def train_test_split(data, test_size=0.3, shuffle=True, random_state=2020):
    if shuffle:
        data = reset(data, random_state=random_state)
    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test


if __name__ == '__main__':
    import argparse

    # python train.py --lang java --lr 0.001 --batch 32 --gru 64 --dw 128 --epoch 5 --model_type proto --times 5
    #
    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--lang')
    parser.add_argument('--lr')
    parser.add_argument('--batch')
    parser.add_argument('--gru')
    parser.add_argument('--dw')
    parser.add_argument('--epoch')
    parser.add_argument('--model_type')
    parser.add_argument('--times')
    args = parser.parse_args()
    lang = args.lang
    lr = float(args.lr)
    BATCH_SIZE = int(args.batch)
    HIDDEN_DIM = int(args.gru)
    ENCODE_DIM = int(args.dw)
    EPOCHS = int(args.epoch)
    model_type = args.model_type
    times = args.times
    data_root = '../data/'

    from model import BatchProgramCC

    model_name = str(lang) + "_" + str(lr) + "_" + str(BATCH_SIZE) + "_" + str(HIDDEN_DIM) + "_" + str(
        ENCODE_DIM) + "_" + str(times)

    print("Train for %s" % model_name)
    data = pd.read_pickle(data_root + lang + '/data_all_blocks.pkl')

    train_data, test_data = train_test_split(data, test_size=0.3, shuffle=True)
    print("Data loaded.")
    word2vec = Word2Vec.load(data_root + lang + "/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0
    print("Word embedding model loaded. The dimension of word vector is %s" % str(ENCODE_DIM))
    USE_GPU = True
    if lang == 'java':
        LABELS = 6
        target_names = ['0', '1', '2', '3', '4', '5']
    elif lang == 'c':
        LABELS = 2
        target_names = ['0', '1']

    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings)
    # model.load_state_dict(torch.load('../model/java_model_tcn_0.01_32_64_128_4.pth'))
    if USE_GPU:
        print("Using GPU device.")
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # training procedure
    print('Start training...')
    print(count_param(model))

    draw_index = 0
    train_loss = []
    viz = Visdom()
    viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    start_time = time.time()
    len_train = len(train_data)
    end_index_train = int(len_train / BATCH_SIZE) - 1
    for epoch in range(EPOCHS):
        index = 0
        while index < end_index_train:
            batch = get_batch(train_data, index, BATCH_SIZE)
            index += 1
            draw_index += BATCH_SIZE
            train1_inputs, train2_inputs, train_labels = batch
            if USE_GPU:
                train1_inputs, train2_inputs, train_labels = train1_inputs, train2_inputs, torch.tensor(train_labels,
                                                                                                        dtype=torch.int64).cuda()
            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train1_inputs, train2_inputs)
            loss = loss_function(output, Variable(train_labels))
            viz.line([loss.item()], [draw_index], win='train_loss', update='append')
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
    end_time = time.time()
    print(end_time - start_time)

    torch.save(model.state_dict(), "../model/" + model_name + ".pth")

    # testing procedure
    predicts = []
    trues = []
    test_loss = 0.0
    i = 0
    len_test = len(test_data)
    end_index_test = int(len_test / BATCH_SIZE) - 1
    while i < end_index_test:
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += 1
        test1_inputs, test2_inputs, test_labels = batch
        if USE_GPU:
            test_labels = torch.tensor(test_labels, dtype=torch.int64).cuda()

        model.batch_size = len(test_labels)
        # model.hidden = model.init_hidden()
        output = model(test1_inputs, test2_inputs)

        loss = loss_function(output, Variable(test_labels))
        # viz.line([loss.item()], [i], win='test_loss', update='append')
        # calc testing acc
        predicts.extend(output.data.cpu().numpy())
        trues.extend(test_labels.cpu().numpy())

        test_loss += loss.item() * len(test_labels)

    predicts = [[i.argmax()] for i in predicts]
    print(classification_report(trues, predicts, target_names=target_names, digits=6))
    train_loss_df = pd.DataFrame({'loss': train_loss})
    train_loss_df.to_pickle("../loss/" + model_name + '_loss.pkl')
