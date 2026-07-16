import random

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import torch.nn.functional as F


class AE(nn.Module):

    def __init__(self, n_enc, hidden,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_in = Linear(n_input, n_enc)
        self.hidden_enc = nn.ModuleList([Linear(n_enc, n_enc) for i in range(hidden)])
        self.z_layer = Linear(n_enc, n_z)

        self.dec_in = Linear(n_z, n_enc)
        self.hidden_dec = nn.ModuleList([Linear(n_enc, n_enc) for i in range(hidden)])
        self.x_bar_layer = Linear(n_enc, n_input)

    def forward(self, x):
        enc_result = []
        enc_result.append(F.relu(self.enc_in(x)))
        for layer in self.hidden_enc:
            enc_result.append(F.relu(layer(enc_result[-1])))
        z = self.z_layer(enc_result[-1])

        dec = F.relu(self.dec_in(z))
        for layer in self.hidden_dec:
            dec = F.relu(layer(dec))
        x_bar = self.x_bar_layer(dec)

        return x_bar, enc_result , z


def pretrain_ae(model, x, n_clusters, epochs, name=None, root_path=None):
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        x_bar, _ = model(x)
        loss = F.mse_loss(x_bar, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # root_path, _ = os.path.split(os.path.abspath(__file__))
    if name is not None:
        torch.save(model.state_dict(),  f'./model/ae_pre_train_{name}.pkl')
    else:
        torch.save(model.state_dict(),  './model/ae_pre_train.pkl')
    torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def pre_train(X, n_clusters, n_input, n_z, n_enc, hidden, pre_ae_epoch, name=None):
    setup_seed(10)
    model = AE(
        n_enc=n_enc,
        hidden=hidden,
        n_input=n_input,
        n_z=n_z).cuda()

    pretrain_ae(model, X, n_clusters, pre_ae_epoch, name)

