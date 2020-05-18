import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
from utils.BBBlayers import *
from torch.utils import data


class OneD_Stack(nn.Module):
    def __init__(self, kernel_size, feature_maps, padding):
        super(OneD_Stack, self).__init__()
        self.kernel_size = kernel_size
        self.feature_maps = feature_maps
        self.padding = padding

        self.conv1 = BBBConv1d(1, self.feature_maps ** 1, kernel_size=self.kernel_size,
                               padding=self.padding)  # same padding 2P = K-1
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = BBBConv1d(self.feature_maps ** 1, self.feature_maps ** 2, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = BBBConv1d(self.feature_maps ** 2, self.feature_maps ** 3, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool1d(2, 2)
        self.flatten = FlattenLayer(250 * 8)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                  self.pool3,
                  self.flatten]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        return x, kl


class Linear_Layers(nn.Module):
    def __init__(self):
        super(Linear_Layers, self).__init__()
        self.fc1 = BBBLinearFactorial(4000, 512)
        self.soft1 = nn.Softplus()
        self.fc2 = BBBLinearFactorial(512, 64)
        self.soft2 = nn.Softplus()
        self.fc3 = BBBLinearFactorial(64, 2)

    def forward(self, x, input_dnu):
        kl = 0
        x, _kl1 = self.fc1.fcprobforward(x)
        x = self.soft1(x)
        x, _kl2 = self.fc2.fcprobforward(x)
        x = self.soft2(x)
        x = torch.add(x, input_dnu.view(-1, 1))
        x, _kl3 = self.fc3.fcprobforward(x)
        kl = kl + _kl1 + _kl2 + _kl3

        return x, kl


class Bayes_Classifier(nn.Module):
    '''The architecture of SLOSH with Bayesian Layers'''

    def __init__(self):
        super(Bayes_Classifier, self).__init__()
        self.conv_stack1 = OneD_Stack(kernel_size=31, feature_maps=2, padding=15)
        self.conv_stack2 = OneD_Stack(kernel_size=31, feature_maps=2, padding=15)
        self.linear = Linear_Layers()
        layers = [self.conv_stack1, self.conv_stack2, self.linear]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x, input_dnu):  # (N,Cin,L)
        'Forward pass with Bayesian weights'
        x = x.unsqueeze(1)
        kl = 0

        stack1, _kl1 = self.conv_stack1(x)
        stack2, _kl2 = self.conv_stack2(x)
        concat = torch.cat((stack1.view(x.size()[0], -1), stack2.view(x.size()[0], -1)), 1)
        logits, _kl_fc = self.linear(concat, input_dnu)
        kl = kl + _kl1 + _kl2 + _kl_fc
        return logits, kl


def get_beta(m, beta_type='standard', batch_idx=None):  # m is the number of minibatches
    if beta_type == "Blundell":  
        ### Weight Uncertainty in Neural Networks, Blundell et al. (2015), Section 3.4 pg 5:
        # The first few minibatches are heavily influenced by the complexity cost (KL), whilst the later minibatches are largely
        # influenced by the data. At the beginning of learning this is particularly useful as for the first few minibatches
        # changes in the weights due to the data are slight and as more data are seen,
        # data become more influential and the prior less influential
        ###
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    else:  # standard from Graves et al. (2011), weights KL divergence in each minibatch equally
        beta = 1 / m
    return beta


def elbo(out, y, kl, beta):
    loss = F.cross_entropy(out, y)
    return loss + beta * kl

