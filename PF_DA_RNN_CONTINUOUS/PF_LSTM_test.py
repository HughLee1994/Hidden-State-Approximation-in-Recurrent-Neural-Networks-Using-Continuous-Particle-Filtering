import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt


class PFLSTM(nn.Module):
    def __init__(self, num_particles, input_size, hidden_size, continuous=False):
        super(PFLSTM, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.continous = continuous

    def init_hidden_states(self, x):
        h0 = x.data.new(self.num_particles, x.shape[0], self.hidden_size).zero_()
        c0 = x.data.new(self.num_particles, x.shape[0], self.hidden_size).zero_()
        p0 = torch.ones(self.num_particles, x.shape[0], 1)*np.log(1/self.num_particles)

        return h0, c0, p0

    def forward(self, input_data, ):