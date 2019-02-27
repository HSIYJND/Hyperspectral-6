from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch
from data_process import load_mat, prepare_torch_data, prepare_batch_data
from models import FullyConnectedNet, ConvolutionNet
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model_and_predict(model, train_x, test_x, train_y, test_y, lr=0.0002, epochs=10000):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    for t in range(epochs):
        prediction = model(train_x)
        loss = loss_func(prediction, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t % 10 == 0:
            print('epoch {}, train loss: {}'.format(t, loss.data[0]))
    model.eval()
    test_pred = torch.max(model(test_x).cpu(), 1)[1].data.numpy().squeeze()
    acc = accuracy_score(test_pred, test_y.data.numpy())
    return model, acc


def train_model_and_predict_2(model, train_x, test_x, train_y, test_y, lr=0.0002, epochs=10000, batch_size=128):
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            prediction = model(batch_x)
            loss = loss_func(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print('epoch {}, train loss: {}'.format(epoch, loss.data[0]))
    test_pred = torch.max(model(test_x).cpu(), 1)[1].data.numpy().squeeze()
    acc = accuracy_score(test_pred, test_y.data.numpy())
    return model, acc


if __name__ == '__main__':
    random_seed = 1995
    epoch = 50
    lr = 0.01
    test_size = 0.25
    use_gpu = False
    add_axis = False
    data_path = './data2/PaviaU.mat'
    label_path = './data2/PaviaU_gt.mat'

    seed_everything(random_seed)

    train_x, test_x, train_y, test_y = load_mat(data_path, label_path, test_rate=test_size)
    n_feature = train_x.shape[1]
    n_class = len(np.unique(train_y))
    train_x, test_x, train_y, test_y = prepare_batch_data(train_x, test_x, train_y, test_y,
                                                          use_cuda=use_gpu, add_axis=add_axis)

    model = FullyConnectedNet(n_feature=n_feature, n_output=n_class)
    if use_gpu:
        model.cuda()
    print(model)
    _, accuracy = train_model_and_predict_2(model, train_x, test_x, train_y, test_y, lr=lr, epochs=epoch)
    print(accuracy)

