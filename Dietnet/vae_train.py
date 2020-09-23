#  Train VAE on dietnet data

import sys
from pathlib import Path
import os
import h5py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append("Dietnet/")
from Dietnet.make_attributions import load_data, load_model
from Dietnet.helpers import dataset_utils as du
from Dietnet.helpers import model as model
from Dietnet.Interpretability import attribution_manager as am
from Dietnet.helpers import mainloop_utils as mlu
from Dietnet.helpers import log_utils as lu

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


#  Dataset Object
class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, xs, xs_unnormed, ys, samples):
        self.xs = xs #tensor on gpu
        self.xs_unnormed = xs_unnormed
        self.ys = ys #tensor on gpu
        self.samples = samples #np array

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Index can be a number or a list of numbers
        x = self.xs[index]
        xun = self.xs_unnormed[index]
        y = self.ys[index]
        sample = self.samples[index]

        return x, xun, y, sample


#  VAE model definition
class encoder(nn.Module):
    def __init__(self, n_feats, n_hidden1_u, latent_dim, 
                 input_dropout=0., eps=1e-5, incl_bias=True):
        super(encoder, self).__init__()

        # Dropout on input layer
        self.input_dropout = nn.Dropout(p=input_dropout)

        self.bn = nn.BatchNorm1d(num_features=n_hidden1_u, eps=eps)

        # 2nd hidden layer
        self.fc = nn.Linear(n_hidden1_u, n_hidden1_u//2)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden1_u//2, eps=eps)
        
        self.mean_fc = nn.Linear(n_hidden1_u//2, latent_dim)
        self.logvar_fc = nn.Linear(n_hidden1_u//2, latent_dim)
        nn.init.xavier_uniform_(self.mean_fc.weight)
        nn.init.xavier_uniform_(self.logvar_fc.weight)
        nn.init.zeros_(self.mean_fc.bias)
        nn.init.zeros_(self.logvar_fc.bias)

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

        z = F.linear(x, fatLayer_weights, bias=self.fat_bias)
        a = torch.relu(z)
        a = self.bn(a)
        a = self.dropout(a)
        
        z = self.fc(a)
        a = torch.relu(z)
        a = self.bn2(a)

        mu = self.mean_fc(a)
        logvar = self.logvar_fc(a)

        return mu, logvar


class decoder(nn.Module):
    def __init__(self, num_inputs, n_hidden1_u, latent_dim, eps=1e-5, incl_bias=True):
        super(decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, n_hidden1_u//2)
        self.bn = nn.BatchNorm1d(num_features=n_hidden1_u//2, eps=eps)
        self.fc2 = nn.Linear(n_hidden1_u//2, n_hidden1_u)
        self.bn2 = nn.BatchNorm1d(num_features=n_hidden1_u, eps=eps)

        #  bias term for fat layer
        if incl_bias:
            self.fat_bias = nn.Parameter(data=torch.rand(num_inputs), requires_grad=True)
            nn.init.zeros_(self.fat_bias)
        else:
            self.fat_bias = None

        self.bn3 = nn.BatchNorm1d(num_features=num_inputs, eps=eps)

    def forward(self, x, fatLayer_weights):
        z = self.fc(x)
        a = torch.relu(z)
        a = self.bn(a)

        z = self.fc2(a)
        a = torch.relu(z)
        a = self.bn2(a)

        x_hat = F.linear(a, fatLayer_weights, bias=self.fat_bias)
        x_hat = torch.relu(x_hat)
        x_hat = self.bn3(x_hat)

        return torch.sigmoid(x_hat)


class VAE(nn.Module):
    def __init__(self, num_inputs, n_feats, n_hidden_u, param_init, latent_dim, input_dropout=0., eps=1e-5, incl_bias=True):
        super(VAE, self).__init__()
        self.feat_emb_enc = model.Feat_emb_net(n_feats, n_hidden_u, param_init)
        self.feat_emb_dec = model.Feat_emb_net(n_feats, n_hidden_u, param_init)
        self.encoder = encoder(n_feats, n_hidden_u, latent_dim, input_dropout, eps, incl_bias)
        self.decoder = decoder(num_inputs, n_hidden_u, latent_dim, eps, incl_bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, emb, x_batch):

        # Forward pass in auxilliary net
        fatLayer_weights_enc = self.feat_emb_enc(emb)

        #  if we are in dataparallel mode, we add extra dim during dataloading and remove it here
        if len(fatLayer_weights_enc.shape) > 2:
            fatLayer_weights_enc = fatLayer_weights_enc[0]
        fatLayer_weights_enc = torch.transpose(fatLayer_weights_enc, 1, 0)

        fatLayer_weights_dec = self.feat_emb_dec(emb)
        if len(fatLayer_weights_dec.shape) > 2:
            fatLayer_weights_dec = fatLayer_weights_dec[0]

        mu, logvar = self.encoder(x_batch, fatLayer_weights_enc)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Forward pass in decoder net
        dec_params = self.decoder(z, fatLayer_weights_dec)

        return dec_params, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(dec_params, x, mu, logvar):

    # Compute -logP(x|z) where we model x~binomial(n=2, p=dec_params)
    BCE = - torch.cat([(1-dec_params[x == 0])**2, 
                       (dec_params[x == 1])*(1-(dec_params[x == 1])),
                       (dec_params[x == 2])**2]).log().sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

    
seed = 23
which_fold = 0
train_valid_ratio = 0.75
batch_size = 12

exp_path = Path('/home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/Dietnet2/1000G_EXP/EXP01_2020.07')
exp_folder = 'REPRODUCE_2020.07'
full_path = exp_path / exp_folder / '{}_fold{}'.format(exp_folder, which_fold)
model_path =  full_path / 'model_params.pt'

dataset = 'dataset.npz'
embedding = 'embedding.npz'
folds_indexes = 'folds_indexes.npz'

# Set GPU
print('Cuda available:', torch.cuda.is_available())
print('Current cuda device ', torch.cuda.current_device())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Fix seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
print('Seed:', str(seed))

# Get fold data (indexes and samples are np arrays, x,y are tensors)
data = du.load_data(os.path.join(exp_path, dataset))
folds_indexes = du.load_folds_indexes(
        os.path.join(exp_path, folds_indexes))
(train_indexes, valid_indexes, test_indexes,
 x_train, y_train, samples_train,
 x_valid, y_valid, samples_valid,
 x_test, y_test, samples_test) = du.get_fold_data(which_fold,
                                                  folds_indexes,
                                                  data,
                                                  split_ratio=train_valid_ratio,
                                                  seed=seed)

# Put data on GPU
x_train, x_valid, x_test = x_train.float(), x_valid.float(), x_test.float()

# Compute mean and sd of training set for normalization
mus, sigmas = du.compute_norm_values(x_train)

# Replace missing values
du.replace_missing_values(x_train, mus)
du.replace_missing_values(x_valid, mus)
du.replace_missing_values(x_test, mus)

# Normalize
x_train_normed = du.normalize(x_train, mus, sigmas)
x_valid_normed = du.normalize(x_valid, mus, sigmas)
x_test_normed = du.normalize(x_test, mus, sigmas)

# Make fold final dataset
train_set = FoldDataset(x_train_normed, x_train, y_train, samples_train)
valid_set = FoldDataset(x_valid_normed, x_valid, y_valid, samples_valid)
test_set = FoldDataset(x_test_normed, x_test, y_test, samples_test)

# Load embedding
emb = du.load_embedding(os.path.join(exp_path, embedding), which_fold)
emb = emb.to(device)
emb = emb.float()

# Normalize embedding
emb_norm = (emb ** 2).sum(0) ** 0.5
emb = emb/emb_norm



# Instantiate model
# Input size
n_feats_emb = emb.size()[1] # input of aux net
n_feats = emb.size()[0] # input of main net

# Hidden layers size
emb_n_hidden_u = 512

#  latent space size
hidden_dim = 100

input_dropout = 0

# Output layer
n_targets = max(torch.max(train_set.ys).item(),
                torch.max(valid_set.ys).item(),
                torch.max(test_set.ys).item()) + 1 #0-based encoding
    
vae = VAE(n_feats, n_feats_emb, emb_n_hidden_u, None, hidden_dim)
vae = nn.DataParallel(vae)
vae = vae.to(device)

print('\n***Nb features in models***')
print('n_feats_emb:', n_feats_emb)
print('n_feats:', n_feats)


# Loss
criterion = loss_function

# Optimizer
lr = 1e-3
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

# Training loop hyper param
n_epochs = 50
batch_size = 138

# Minibatch generators
train_generator = DataLoader(train_set, batch_size=batch_size)
valid_generator = DataLoader(valid_set,
                             batch_size=batch_size,
                             shuffle=False)
test_generator = DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False)

#  prepare for dataparallel mode (by duplicating along 0th dimension, so each GPU gets a copy!)
emb = emb.unsqueeze(0).repeat(repeats=(2,1,1)).to(device)


losses = []
bces = []
klds = []

for e in range(n_epochs):
    for x_batch, x_batch_unnormed, _, _ in train_generator:
        vae = vae.train()
        optimizer.zero_grad()

        # Forward pass
        dec_params, mu, logvar = vae(emb, x_batch.to(device))

        # Compute loss
        loss, bce, kld = criterion(dec_params, x_batch_unnormed, mu, logvar)

        # Compute gradients in discrim net
        loss.backward()

        # Optim
        optimizer.step()

        # Monitoring: Minibatch
        losses.append(loss.item())
        bces.append(bce.item())
        klds.append(kld.item())
        print('[{}/{}] loss: {:.3f} bce: {:.3f} kld: {:.3f}'.format(len(losses)%len(train_generator), 
                                                                    len(train_generator), 
                                                                    loss.item(), 
                                                                    bce.item(),
                                                                    kld.item()))
    with torch.no_grad():
        val_loss, val_bce, val_kld, accs = 0, 0, 0, 0
        for x_batch, x_batch_unnormed, _, _ in valid_generator:
            vae = vae.eval()

            # Forward pass
            dec_params, mu, logvar = vae(emb, x_batch.to(device))

            #  sample from decoder posterior (binomial with support [0,1,2])
            dec_dist = torch.distributions.binomial.Binomial(total_count=2, probs=dec_params)
            out_sample = dec_dist.sample()

            # Compute loss
            crit_out = criterion(dec_params, x_batch_unnormed, mu, logvar)
            val_loss += crit_out[0].item()
            val_bce += crit_out[1].item()
            val_kld += crit_out[2].item()
            accs += (out_sample.cpu() == x_batch_unnormed).float().sum().item()

        # Monitoring: Minibatch
        print('[epoch {}] loss: {:.3f} bce: {:.3f} kld: {:.3f} acc: {:.3f}'.format(e, 
                                                                                   val_loss/len(valid_generator), 
                                                                                   val_bce/len(valid_generator), 
                                                                                   val_kld/len(valid_generator),
                                                                                   accs/(out_sample.shape[0]*out_sample.shape[1]*len(valid_generator))))

    print('completed epoch: {}/{}'.format(e, n_epochs))
