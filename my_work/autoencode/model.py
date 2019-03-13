import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm


class AutoEncoder(nn.Module):
    def __init__(self, n_bands):
        super(AutoEncoder, self).__init__()
        self.n_bands = n_bands
        self.encoder = nn.Sequential(
            nn.Linear(self.n_bands, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_bands),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model, train_data, lr, epochs, use_cuda=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    train_x = Variable(torch.FloatTensor(train_data))
    losses = []
    if use_cuda:
        train_x = train_x.cuda()
        model.cuda()
    model.train()
    for _ in tqdm(range(epochs)):
        decoded = model(train_x)
        loss = loss_func(decoded, train_x)
        losses.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, decoded, losses


if __name__ == '__main__':
    test_model = AutoEncoder(200)
    print(test_model)
