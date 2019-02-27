from torch import nn
import torch.nn.functional as F
import torch


class FullyConnectedNet(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(FullyConnectedNet, self).__init__()
        self.hidden_size_1 = 512
        self.hidden_size_2 = 320
        self.hidden_size_3 = 128

        self.hidden_1 = nn.Linear(n_feature, self.hidden_size_1)
        self.bn_1 = nn.BatchNorm1d(self.hidden_size_1)
        self.drop_1 = nn.Dropout(p=0.25)
        self.hidden_2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.bn_2 = nn.BatchNorm1d(self.hidden_size_2)
        self.hidden_3 = nn.Linear(self.hidden_size_2, self.hidden_size_3)
        self.bn_3 = nn.BatchNorm1d(self.hidden_size_3)
        self.out = nn.Linear(self.hidden_size_3, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = self.bn_1(x)
        x = self.drop_1(x)
        x = F.relu(self.hidden_2(x))
        x = self.bn_2(x)
        x = F.relu(self.hidden_3(x))
        x = self.bn_3(x)
        x = self.out(x)
        return x


class ConvolutionNet(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(ConvolutionNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.hidden_1 = nn.Linear(n_feature*8//2, n_feature)
        self.hidden_2 = nn.Linear(n_feature, 100)
        self.hidden_3 = nn.Linear(100, n_output)

    def forward(self, x):
        x = self.conv_1(x)
        x = x.view(x.size(0), -1)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Generator, self).__init__()
        self.hidden_size_1 = 64
        self.hidden_size_2 = 64

        self.hidden_1 = nn.Linear(n_feature, self.hidden_size_1)
        self.hidden_2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.output = nn.Linear(self.hidden_size_2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Discriminator, self).__init__()
        self.hidden_size_1 = 64
        self.hidden_size_2 = 64

        self.hidden_1 = nn.Linear(n_feature, self.hidden_size_1)
        self.hidden_2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.output = nn.Linear(self.hidden_size_2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.sigmoid(self.output(x))
        return x


if __name__ == '__main__':
    n_feature = 200
    n_output = 16
    fcnn = FullyConnectedNet(n_feature, n_output)
    print('==='*10)
    print('fully connected net:\n', fcnn)
    conv = ConvolutionNet(n_feature, n_output)
    print('===' * 10)
    print('\nconv_1d net:\n', conv)
    generator_nn = Generator(n_feature, n_output)
    print('===' * 10)
    print('\nGenerator net:\n', generator_nn)
    discriminator_nn = Discriminator(n_feature, n_output)
    print('===' * 10)
    print('\nDiscriminator net:\n', discriminator_nn)
