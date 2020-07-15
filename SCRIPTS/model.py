import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Feat_emb_net(nn.Module):
    def __init__(self, n_feats, n_hidden_u, param_init):
        super(Feat_emb_net, self).__init__()

        # Theano values for param init
        if param_init is not None:
            params = np.load(param_init)

        # 1st hidden layer
        self.hidden_1 = nn.Linear(n_feats, n_hidden_u, bias=False)
        if param_init is not None:
            self.hidden_1.weight = torch.nn.Parameter(
                    torch.from_numpy(params['w1_aux']))
        else:
            nn.init.uniform_(self.hidden_1.weight, a=-0.02, b=0.02)

        # 2nd hidden layer
        self.hidden_2 = nn.Linear(n_hidden_u, n_hidden_u, bias=False)
        if param_init is not None:
            self.hidden_2.weight = torch.nn.Parameter(
                    torch.from_numpy(params['w2_aux']))
        else:
            nn.init.uniform_(self.hidden_2.weight, a=-0.02, b=0.02)


        self.hidden_2 = nn.Linear(n_hidden_u, n_hidden_u, bias=False)
        nn.init.uniform_(self.hidden_2.weight, a=-0.02, b=0.02)

    def forward(self, x):
        ze1 = self.hidden_1(x)
        ae1 = torch.tanh(ze1)
        ze2 = self.hidden_2(ae1)
        ae2 = torch.tanh(ze2)

        return ae2


class Discrim_net(nn.Module):
    def __init__(self, fatLayer_weights, n_feats,
                 n_hidden1_u, n_hidden2_u, n_targets,
                 input_dropout = 0.):
        super(Discrim_net, self).__init__()

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)

        # 1st hidden layer
        self.hidden_1 = nn.Linear(n_feats, n_hidden1_u)
        self.hidden_1.weight = torch.nn.Parameter(fatLayer_weights)
        nn.init.zeros_(self.hidden_1.bias)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1_u)

        # 2nd hidden layer
        self.hidden_2 = nn.Linear(n_hidden1_u, n_hidden2_u)
        nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.zeros_(self.hidden_2.bias)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden2_u)

        # Output layer
        self.out = nn.Linear(n_hidden2_u, n_targets)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Dropout
        self.dropout = nn.Dropout()


    def forward(self, x):
        # input size: batch_size x n_feats
        x = self.input_dropout(x)

        z1 = self.hidden_1(x)
        a1 = torch.relu(z1)
        a1 = self.bn1(a1)
        a1 = self.dropout(a1)

        z2 = self.hidden_2(a1)
        a2 = torch.relu(z2)
        a2 = self.bn2(a2)
        a2 = self.dropout(a2)

        out = self.out(a2)
        # Softmax will be computed in the loss

        return out


class Discrim_net2(nn.Module):
    """
    Discrim_net modified to take fatLayer_weights as a forward arg.
    Does not have weights for first layer;
    Uses F.linear with passed weights instead
    """
    def __init__(self, n_feats,
                 n_hidden1_u, n_hidden2_u, n_targets,
                 param_init, input_dropout = 0., incl_bias=True):
        super(Discrim_net2, self).__init__()

        # Theano values for params init
        if param_init is not None:
            params = np.load(param_init)

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)

        # 1st hidden layer (we don't need this anymore)
        #self.hidden_1 = F.linear(input, weight, bias=None)
        self.bn1 = nn.BatchNorm1d(num_features=n_hidden1_u)

        # 2nd hidden layer
        self.hidden_2 = nn.Linear(n_hidden1_u, n_hidden2_u)
        if param_init is not None:
            self.hidden_2.weight = torch.nn.Parameter(
                    torch.from_numpy(params['w2_main']))
        else:
            nn.init.xavier_uniform_(self.hidden_2.weight)
        nn.init.zeros_(self.hidden_2.bias)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden2_u)

        # Output layer
        self.out = nn.Linear(n_hidden2_u, n_targets)
        if param_init is not None:
            self.out.weight = torch.nn.Parameter(
                    torch.from_numpy(params['w3_main']))
        else:
            nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        #  bias term for fat layer
        if incl_bias:
            self.fat_bias = nn.Parameter(data=torch.rand(n_hidden1_u), requires_grad=True)
            nn.init.zeros_(self.fat_bias)
        else:
            self.fat_bias = None

        # Dropout
        self.dropout = nn.Dropout()


    def forward(self, x, fatLayer_weights):
        # input size: batch_size x n_feats
        # weight = comes from feat embedding net
        # now ^^^ is passed with forward
        x = self.input_dropout(x)

        z1 = F.linear(x, fatLayer_weights, bias=self.fat_bias)
        #z1 = self.hidden_1(x)
        a1 = torch.relu(z1)
        a1 = self.bn1(a1)
        a1 = self.dropout(a1)

        z2 = self.hidden_2(a1)
        a2 = torch.relu(z2)
        a2 = self.bn2(a2)
        a2 = self.dropout(a2)

        out = self.out(a2)
        # Softmax will be computed in the loss

        return out


class CombinedModel(nn.Module):
    def __init__(self, n_feats, n_hidden_u, n_hidden1_u, n_hidden2_u,
                 n_targets, param_init, input_dropout=0., incl_bias=True):
        super(CombinedModel, self).__init__()

        # Initialize feat. embedding and discriminative networks
        self.feat_emb = Feat_emb_net(n_feats, n_hidden_u, param_init)
        self.disc_net = Discrim_net2(n_feats, n_hidden1_u, n_hidden2_u,
                                     n_targets, param_init, input_dropout, incl_bias)


    def forward(self, emb, x_batch):
        # Forward pass in auxilliary net
        feat_emb_model_out = self.feat_emb(emb)
        # Forward pass in discrim net
        fatLayer_weights = torch.transpose(feat_emb_model_out,1,0)
        discrim_model_out = self.disc_net(x_batch, fatLayer_weights)

        return discrim_model_out


if __name__ == '__main__':
    # Let's do a little test just to see if the run fails
    # Dummy data
    x = torch.ones((30,3000),requires_grad=True)
    y = torch.LongTensor(30).random_(0, 2)
    x_emb = torch.ones((3000,3*3), requires_grad=True)

    # Intantiate models
    emb_model = Feat_emb_net(n_feats=x_emb.size()[1], n_hidden_u=100)
    emb_model_out = emb_model(x_emb)
    print(emb_model_out.size())
    fatLayer_weights = torch.transpose(emb_model_out,1,0)
    discrim_model = Discrim_net(fatLayer_weights=fatLayer_weights,
                                n_feats=x.size()[1],
                                n_hidden1_u=100, n_hidden2_u=100,
                                n_targets=y.size()[0])
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    params = list(discrim_model.parameters()) + list(emb_model.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1)

    # Training loop
    for epoch in range(10):
        print('epoch:', epoch)
        optimizer.zero_grad()

        # Forward pass in feat emb net
        emb_out = emb_model(x_emb)

        # Set fat layer weights in discrim net
        fatLayer_weights = torch.transpose(emb_out,1,0)
        discrim_model.hidden_1.weight.data = fatLayer_weights

        # Forward pass in discrim net
        discrim_out = discrim_model(x)

        # Compute loss
        loss = criterion(discrim_out, y)
        print('Loss:', loss)

        # Compute gradients in discrim net
        loss.backward()
        # Copy weights of W1 in discrim net to emb_out.T
        fatLayer_weights.grad = discrim_model.hidden_1.weight.grad
        # Compute gradients in feat. emb. net
        torch.autograd.backward(fatLayer_weights, fatLayer_weights.grad)

        # Optim
        optimizer.step()
