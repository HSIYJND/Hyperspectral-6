import numpy as np
from data_process import load_mat
from torch.autograd import Variable
import torch.utils.data as Data
import torch


def train_gan(train_x, test_x, train_y, test_y, G, D, LR_G, LR_D, n_ideas=32, batch_size=128, epochs=10000):
    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
    train_dataset = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    G.train()
    D.train()
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            G_ideas = Variable(torch.randn(batch_size, n_ideas), requires_grad=True)
            G_output = G(G_ideas)

            prob_0 = D(batch_x)
            prod_1 = D(G_output)

            D_loss = - torch.mean(torch.log(prob_0)) + torch.log(1. - prod_1)
            G_loss = torch.mean(torch.log(1. - prod_1))

            opt_D.zero_grad()
            D_loss.backward(retain_variables=True)
            opt_D.step()

            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()
    D.eval()
    test_pre = D(test_x)

    return D, test_pre


if __name__ == '__main__':
    epoch = 100
    lr = 0.01
    test_size = 0.25
    use_gpu = False
    add_axis = False
    data_path = './data2/PaviaU.mat'
    label_path = './data2/PaviaU_gt.mat'

    train_x, test_x, train_y, test_y = load_mat(data_path, label_path, test_rate=test_size)
    n_feature = train_x.shape[1]
    n_class = len(np.unique(train_y))


