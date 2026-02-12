import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

from utils import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser(description="Particle Filter RNNs,")
    parser.add_argument('--dataroot', type=str, default='data/nasdaq100_padding9.csv', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size 128')

    parser.add_argument('--num_particles', type=int, default=50, help='number of particles in particle filter')
    parser.add_argument('--encoder_hidden_size', type=int, default=128, help='encoder hidden size m[64, 128]')
    parser.add_argument('--decoder_hidden_size', type=int, default=128, help='decoder hidden size p[64, 128]')
    parser.add_argument('--timesteps', type=int, default=10, help='number of time steps in the window T[10]')

    parser.add_argument('--epochs', type=int, default=220, help='number of epochs to train [10, 100, 1000]')
    parser.add_argument('--lr', type=float, default=0.0011, help='learning rate')

    parser.add_argument('--encoder_pf', type=bool, default=True, help='use particle filter in the Encoder')
    parser.add_argument('--decoder_pf', type=bool, default=True, help='use particle filter in the Decoder')
    parser.add_argument('--parallel', type=bool, default=True, help='parallel optimization of the Encoder and Decoder')

    args = parser.parse_args()

    return args


def main():
    torch.manual_seed(9)
    args = parse_args()
    print('read the dataset ...')
    X, y = read_data(args.dataroot, debug=False)

    print('initialize model ...')

    model = DA_RNN(X, y, args.timesteps, args.num_particles, args.encoder_hidden_size, args.decoder_hidden_size, args.batch_size, args.lr, args.epochs, args.encoder_pf,
                   args.decoder_pf, args.parallel)

    print("Training the model ...")
    model.train()

    print("Testing the model")
    y_pred = model.test()
    print('y_pred', y_pred.shape)
    y_gt = model.y[model.train_timesteps:]
    print('ground truth:', len(y_gt))
    # MSE, MAE, RMSE
    error = []
    abs_error = []
    for i in range(len(y_gt)):
        error.append(y_gt[i] - y_pred[i])
        abs_error.append(abs(y_gt[i] - y_pred[i])/abs(y_gt[i]))
    print('error', len(error))
    squareError = []
    absError = []
    for val in error:
        squareError.append(val*val)
        absError.append(abs(val))
    MSE = sum(squareError)/len(squareError)
    print('length of target', len(squareError))
    from math import sqrt
    RMSE = sqrt(sum(squareError)/len(squareError))
    MAE = sum(absError)/len(absError)
    MAPE = sum(absError/abs(y_gt))/len(absError)
    MAPE2 = sum(abs_error)/len(abs_error)
    print('MSE:', MSE)
    print('RMSE:', RMSE)
    print('MAE:', MAE)
    print('MAPE:', MAPE)
    print('MAPE2:', MAPE2)



    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("iter_losses.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("epoch_losses.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_timesteps:], label='True')
    plt.legend(loc='upper left')
    plt.savefig("predicted.png")
    plt.close(fig3)
    print("Training finished")


if __name__ == '__main__':
    main()