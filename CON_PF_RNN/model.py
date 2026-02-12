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

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def init_hidden_states(self, x):
        h0 = x.data.new(self.num_particles, x.shape[0], self.hidden_size).zero_()
        c0 = x.data.new(self.num_particles, x.shape[0], self.hidden_size).zero_()
        p0 = torch.ones(self.num_particles, x.shape[0], 1)*np.log(1/self.num_particles)

        return h0, c0, p0

    def forward(self, input_data, hiddens, cells):
        batch_size = hiddens.size(0)/self.num_particles
        self.lstm_layer.flatten_parameters()
        _, states = self.lstm_layer(input_data.unsqueeze(0), (hiddens, cells))
        hiddens = states[0]
        cells = states[1]

        return hiddens, cells


def init_hidden_(x, hidden_size):
    return torch.zeros(1, x.size(0), hidden_size)


def reparameterize_(mu, var):
    """
    Reparameterization trick

    :param mu: mean
    :param var: variance
    :return: new samples from the Gaussian distribution
    """
    std = torch.nn.functional.softplus(var)
    # if torch.cuda.is_available():
    #     eps = torch.cuda.FloatTensor(std.shape).normal_()
    # else:
    #     eps = torch.FloatTensor(std.shape).normal_()
    eps = torch.FloatTensor(std.shape).normal_()
    return mu + eps * std


def resampling_(num_particles, particles, prob):
    """
    The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

    :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                    each tensor has a shape: [num_particles * batch_size, h_dim]
    :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
    :return: resampled particles and weights according to soft-resampling scheme.
    """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    prob = prob.view(num_particles, -1)
    indices = torch.multinomial(prob.transpose(0, 1), num_samples=num_particles, replacement=True)
    batch_size = indices.size(0)

    indices = indices.transpose(1, 0).contiguous()
    # offset = torch.cuda.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
    offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
    # if torch.cuda.is_available():
    #     offset = offset.cuda()
    indices = offset + indices * batch_size
    flatten_indices = indices.view(-1, 1).squeeze()
    # print(flatten_indices)

    # PFLSTM
    particles_new = (particles[0][flatten_indices],
                     particles[1][flatten_indices])

    # prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
    # prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
    #                                                        self.resamp_alpha) / self.num_particles)
    # prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
    # prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
    # prob_new = prob_new.view(-1, 1)
    # prob = torch.ones(batch_size * indices.size(1), 1) * np.log(1 / self.num_particles)

    return particles_new


def sort_hidden(batch_size, num_particles, hiddens_proj, hiddens, hidden_size):
    hiddens_proj = hiddens_proj.view(num_particles, -1).transpose(1, 0).contiguous()

    indices = torch.argsort(hiddens_proj, dim=1)
    # print('indices', indices, indices.shape)
    offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(1)
    # if torch.cuda.is_available():
    #     offset = offset.cuda()
    # indices= indices+offset*num_particles
    indices = offset + indices * batch_size

    flatten_indices = indices.transpose(1, 0).contiguous().view(-1, 1).squeeze()
    hiddens = hiddens.view(-1, hidden_size)[flatten_indices]

    return hiddens


class Predict(nn.Module):
    def __init__(self, num_particles, input_size, hidden_size, T, device, pf, feats=1):
        super(Predict, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.device = device
        self.pf = pf
        self.feats = feats

        self.var_layer = nn.Linear(in_features=hidden_size+input_size+self.feats, out_features=hidden_size)
        self.obs_extractor = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU()
        )

        self.lstm_layer = nn.LSTM(self.input_size+self.feats, hidden_size)
        self.pdf_layer = nn.Linear(in_features=hidden_size+input_size+self.feats, out_features=1)

        self.fc_final = nn.Linear(hidden_size, feats)

    def init_hidden(self, x):
        return x.data.new(1, x.shape[0], self.hidden_size).zero_()

    def reparameterize(self, mu, var):
        std = torch.nn.functional.softplus(var)
        eps = torch.FloatTensor(std.shape).normal_()
        return mu + eps * std

    def resampling(self, num_particles, particles, prob, device):
        prob = prob.view(num_particles, -1)
        indices = torch.multinomial(prob.transpose(0, 1), num_samples=num_particles, replacement=True).to(device)
        batch_size = indices.size(0)

        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0).to(device)
        indices = (offset + indices * batch_size)
        flatten_indices = indices.view(-1, 1).squeeze()

        particles_new = (particles[0][flatten_indices],
                         particles[1][flatten_indices])

        return particles_new

    def forward(self, input_data, y_prev):
        weights = 0
        if self.pf:
            weights = 0
            hiddens = self.init_hidden(input_data).repeat(1, self.num_particles, 1)
            cells = self.init_hidden(input_data).repeat(1, self.num_particles, 1)

            batch_size = input_data.shape[0]
            for t in range(self.T):
                hidden = torch.mean(hiddens.view(self.num_particles, -1, self.hidden_size), dim=0).unsqueeze(0)
                cell = torch.mean(cells.view(self.num_particles, -1, self.hidden_size), dim=0).unsqueeze(0)
                new_input = torch.cat((input_data, y_prev.unsqueeze(2)), dim=2)
                x = torch.cat((hidden.repeat(self.input_size+self.feats, 1, 1).permute(1, 0, 2),
                               cell.repeat(self.input_size+self.feats, 1, 1).permute(1, 0, 2),
                               new_input.permute(0, 2, 1)), dim=2)

                new_input2 = new_input[:, t, :]
                new_input2 = new_input2.squeeze()
                x = torch.cat((new_input2, hidden.squeeze()), dim=1)
                var = self.var_layer(x)  # (batch,encoder_hidden_size, 1)
                self.lstm_layer.flatten_parameters()

                _, states = self.lstm_layer(new_input2.unsqueeze(0).repeat(1, self.num_particles, 1), (
                hiddens.view(1, -1, self.hidden_size), cells.view(1, -1, self.hidden_size)))



                hiddens = states[0]
                hiddens = self.reparameterize(hiddens, var.unsqueeze(0).repeat(1, self.num_particles, 1))
                cells = states[1]
                hiddens_proj = self.fc_final(hiddens.view(-1, self.hidden_size))
                hiddens = sort_hidden(batch_size, self.num_particles, hiddens_proj, hiddens, self.hidden_size)
                y = torch.cat((hiddens.view(-1, self.hidden_size),
                               new_input2.repeat(self.num_particles, 1)), dim=1)
                logpdf_obs = self.pdf_layer(y)
                prob = logpdf_obs.view(-1, self.num_particles, 1)
                prob = torch.exp(prob)
                weights += torch.mean(prob, dim=1)
                prob = prob/torch.sum(prob, dim=1, keepdim=True)

            y_pred = self.fc_final(torch.mean(hiddens.view(self.num_particles, -1, self.hidden_size), dim=0).squeeze())
            return y_pred, weights
        else:
            hidden = self.init_hidden(input_data)
            cell = self.init_hidden(input_data)
            for t in range(self.T):

                new_input = torch.cat((input_data, y_prev.unsqueeze(2)), dim=2)
                new_input2 = new_input[:, t, :]
                new_input2 = new_input2.squeeze()

                self.lstm_layer.flatten_parameters()

                _, states = self.lstm_layer(new_input2.unsqueeze(0), (hidden, cell))
                hidden = states[0]
                cell = states[1]
            y_pred = self.fc_final(hidden[0, :, :])
            return y_pred, weights



class DA_RNN(nn.Module):
    def __init__(self, X, y, T, num_particles, hidden_size, batch, lr, epochs, pf, parallel):
        super(DA_RNN, self).__init__()
        self.num_particles = num_particles
        self.hidden_size = hidden_size
        self.T = T
        self.X = X
        self.y = y
        self.batch = batch
        self.lr = lr
        self.epochs = epochs
        self.parallel = parallel

        self.pf = pf
        self.shuffle = False

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print('run on :', self.device)

        self.Predict = Predict(num_particles=num_particles, input_size=X.shape[1], hidden_size=hidden_size, T=T, device=self.device, pf=pf, feats=1)
        self.criterion = nn.MSELoss()

        self.predict_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.Predict.parameters()),
                                            lr=self.lr)
        self.train_timesteps = int(self.X.shape[0]*0.9)
        self.y = self.y - np.mean(self.y[:self.train_timesteps])

        self.input_size = self.X.shape[1]

    def train(self):
        iter_per_epoch = int(np.ceil(self.train_timesteps*1./self.batch))
        self.iter_losses = np.zeros(self.epochs*iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_tier = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                indices = ref_idx[idx:(idx+self.batch)]
                x = np.zeros((len(indices), self.T, self.input_size))
                y_prev = np.zeros((len(indices), self.T))

                y_gt = self.y[indices + self.T + 1]   # may have some problem maybe y_gt = self.y[indices + self.T]

                for i in range(len(indices)):
                    x[i, :, :] = self.X[indices[i]: (indices[i]+self.T), :]
                    y_prev[i, :] = self.y[indices[i]: (indices[i]+self.T)]

                loss = self.train_forward(x, y_prev, y_gt)
                self.iter_losses[int(epoch*iter_per_epoch + idx/self.batch)] = loss

                idx += self.batch
                n_tier += 1

                if n_tier % 100 == 0 and n_tier != 0:
                    for param_group in self.predict_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.00

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(epoch*iter_per_epoch, (epoch+1)*iter_per_epoch)])

            if epoch % 2 == 0:
                print("Epochs: ", epoch, "Iterations: ", n_tier, "Loss: ",self.epoch_losses[epoch])

            if epoch % 2 == 0:
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)

                if epoch % 400 == 0:
                    y_pred = np.concatenate((y_train_pred, y_test_pred))
                    plt.ioff()
                    plt.figure()
                    plt.plot(range(1, 1+len(self.y)), self.y, label="True")
                    plt.plot(range(self.T, len(y_train_pred)+self.T), y_train_pred, label="Predicted: Train")
                    plt.plot(range(self.T+len(y_train_pred), len(self.y)), y_test_pred, label="Predicted: Test")
                    plt.legend(loc='upper left')
                    plt.show()


    def train_forward(self, X, y_prev, y_gt):
        # self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        self.predict_optimizer.zero_grad()

        # input_weighted, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        # y_pred = self.Decoder(input_encoded, Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_pred, weights = self.Predict(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)),Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(y_gt).type(torch.FloatTensor).to(self.device)).view(-1, 1)

        loss = self.criterion(y_pred, y_true)
        # print('loss', loss.shape, loss)
        loss.backward()

        # self.encoder_optimizer.step()
        # self.decoder_optimizer.step()
        self.predict_optimizer.step()

        return loss.item()

    def test(self, on_train=False):
        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i+self.batch)]
            X = np.zeros((len(batch_idx), self.T, self.X.shape[1]))
            y_prev = np.zeros((len(batch_idx), self.T))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(batch_idx[j], batch_idx[j]+self.T), :]

                    y_prev[j, :] = self.y[range(batch_idx[j], batch_idx[j]+self.T)]
                else:
                    X[j, :, :] = self.X[range(batch_idx[j]+self.train_timesteps-self.T, batch_idx[j]+self.train_timesteps), :]
                    y_prev[j, :] = self.y[range(batch_idx[j]+self.train_timesteps-self.T, batch_idx[j]+self.train_timesteps)]

            y_prev = Variable(torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device))
            X = Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device))

            tmp, _ = self.Predict(X, y_prev)
            y_pred[i: (i + self.batch)] = tmp.cpu().data.numpy()[:, 0]
            # _, input_encoded = self.Encoder(X)
            #
            # y_pred[i: (i+self.batch)] = self.Decoder(input_encoded, y_prev).cpu().data.numpy()[:, 0]

            i += self.batch
        return y_pred















