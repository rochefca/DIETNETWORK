import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from helpers import model


def eval_step(valid_generator, set_size, discrim_model, criterion):
    valid_minibatch_mean_losses = []
    valid_minibatch_n_right = [] # nb of good classifications

    for x_batch, y_batch, _ in valid_generator:
        # Forward pass
        discrim_model_out = discrim_model(x_batch)

        # Predictions
        _, pred = get_predictions(discrim_model_out)

        # Loss
        loss = criterion(discrim_model_out, y_batch)

        # Minibatch monitoring
        weighted_loss = loss.item()*len(y_batch) # for unequal minibatches
        valid_minibatch_mean_losses.append(weighted_loss)
        valid_minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

    valid_loss = np.array(valid_minibatch_mean_losses).sum()/set_size
    valid_acc = compute_accuracy(valid_minibatch_n_right, set_size)

    return valid_loss, valid_acc


def get_predictions(model_output):
    with torch.no_grad():
        score = F.softmax(model_output, dim=1)
        _, pred = torch.max(score, dim=1)

    return score, pred


def compute_accuracy(n_right, set_size):
    acc = np.array(n_right).sum() / float(set_size)*100

    return acc


def has_improved(best_acc, actual_acc, min_loss, actual_loss):
    if actual_acc > best_acc:
        return True
    if actual_acc == best_acc and actual_loss < min_loss:
        return True

    return False


def test(test_generator, set_size, discrim_model):
    test_minibatch_n_right = [] # nb of good classifications in a minibatch

    for i, (x_batch, y_batch, samples) in enumerate(test_generator):
        # Forward pass
        discrim_model_out = discrim_model(x_batch)

        # Predictions
        score, pred = get_predictions(discrim_model_out)
        if i == 0:
            test_pred = pred
            test_score = score
        else:
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

        # Nb of good classifications for the minibatch
        test_minibatch_n_right.append(((y_batch - pred) == 0).sum().item())

    # Total accuracy
    test_acc = compute_accuracy(test_minibatch_n_right, set_size)

    return test_score, test_pred, test_acc
<<<<<<< HEAD
=======


def create_disc_model(comb_model, emb, device):
    """
    this only works with 1 GPU or CPU
    note: the function name is misleading. This does not create a new model. It just
    takes the pre-existing model and
    returns a function that performs the forward pass on it (with fixed embedding).
    this should be okay since python passes args by reference, so even if comb_model
    weights change, the corresponding function will also change
    
    Finally, remember that disc_model will be in whatever mode comb_model is, 
    so if comb_model is in train mode, switch it to eval mode before calling 
    ßthe output of this.
    """

    if torch.cuda.device_count() > 1:
        print('warning: this only works during training/inference with 1 GPU!')
    comb_model = comb_model.eval()
    comb_model.to(device)
    discrim_model = lambda x: comb_model(emb, x) # recreate discrim_model
    return discrim_model


def create_disc_model_multi_gpu(comb_model, emb, device, eps=1e-5, incl_softmax=False):
    """
    Transforms comb_model + emb into equivalent discrim model (with fatlayer weights added as parameters)
    This model can now be sent to multiple GPUs without any bugs
    (cannot do multi-GPU with comb_model since dataparallel will attempt to split the embedding up, 
    which will result in size incompatibilities)
    
    Must pass batchnorm eps seperately in case loading from Theano model!
    Must pass incl_softmax seperately in case loading from Theano model!
    """

    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100
    
    #  put in eval mode and send to correct device
    comb_model = comb_model.eval().to(device)
    emb = emb.to(device)
    fatLayer_weights = torch.transpose(comb_model.feat_emb(emb),1,0)

    #  initialize discriminitive network with fatlayer as a parameter
    #  send back to cpu while loading weights
    disc_net = model.Discrim_net(fatLayer_weights, 
                                 n_feats=n_feats_emb, 
                                 n_hidden1_u=discrim_n_hidden1_u, 
                                 n_hidden2_u=discrim_n_hidden2_u, 
                                 n_targets=26,
                                 eps=eps,
                                 incl_softmax=incl_softmax)

    #  copy over all weights
    disc_net.out.weight.data = comb_model.disc_net.out.weight.data
    disc_net.out.bias.data = comb_model.disc_net.out.bias.data
    disc_net.bn2.weight.data = comb_model.disc_net.bn2.weight.data
    disc_net.bn2.bias.data = comb_model.disc_net.bn2.bias.data
    disc_net.bn2.running_mean = comb_model.disc_net.bn2.running_mean
    disc_net.bn2.running_var = comb_model.disc_net.bn2.running_var
    disc_net.hidden_2.weight.data = comb_model.disc_net.hidden_2.weight.data
    disc_net.hidden_2.bias.data = comb_model.disc_net.hidden_2.bias.data
    disc_net.bn1.weight.data = comb_model.disc_net.bn1.weight.data
    disc_net.bn1.bias.data = comb_model.disc_net.bn1.bias.data
    disc_net.bn1.running_mean = comb_model.disc_net.bn1.running_mean
    disc_net.bn1.running_var = comb_model.disc_net.bn1.running_var
    disc_net.hidden_1.bias.data = comb_model.disc_net.fat_bias.data

    assert (disc_net.hidden_1.weight.data == fatLayer_weights).all()
    disc_net = disc_net.eval().to('cpu')

    #  Now we can do dataparallel
    if torch.cuda.device_count() > 1:
        disc_net = nn.DataParallel(disc_net)
        print("{} GPUs detected! Running in DataParallel mode".format(torch.cuda.device_count()))
    disc_net.to(device)
    return disc_net

def convert_theano_array_to_pytorch_tensor(tensor, array):
    array_as_tensor = torch.from_numpy(array.T)
    assert tensor.data.shape == array_as_tensor.shape
    tensor.data = array_as_tensor

def convert_theano_array_to_pytorch_tensor_1d(tensor, array):
    array_as_tensor = torch.from_numpy(array)
    assert tensor.data.shape == array_as_tensor.shape
    tensor.data = array_as_tensor

def convert_theano_bn_pytorch(bn, theano_arr_1, theano_arr_2, theano_arr_3, theano_arr_4):
    # Idea from: https://discuss.pytorch.org/t/is-there-any-difference-between-theano-convolution-and-pytorch-convolution/10580/10
    # Specifically how to load BN from theano to pytorch
    #extractor += [nn.BatchNorm2d(in_channel)]
    #extractor[-1].weight.data = torch.from_numpy(parameters[i-3])
    #extractor[-1].bias.data = torch.from_numpy(parameters[i-2])
    #extractor[-1].running_mean = torch.from_numpy(parameters[i-1])
    #extractor[-1].running_var = torch.from_numpy((1./(parameters[i]**2)) - 1e-4)

    convert_theano_array_to_pytorch_tensor_1d(bn.weight, theano_arr_2)
    convert_theano_array_to_pytorch_tensor_1d(bn.bias, theano_arr_1)
    convert_theano_array_to_pytorch_tensor_1d(bn.running_mean, theano_arr_3)
    convert_theano_array_to_pytorch_tensor_1d(bn.running_var, (1./(theano_arr_4**2)) - 1e-4)


def load_theano_model(n_feats_emb, emb_n_hidden_u, discrim_n_hidden1_u, discrim_n_hidden2_u, n_targets, theano_weight_file, device, only_discrim_model=True):
    #  loads theano weights file into PyTorch's comb_net OR Discrim_net

    theano_model_params = np.load(theano_weight_file)
    print('theano layer shapes:')
    for f in theano_model_params.files:
        print(f, theano_model_params[f].shape)

    comb_model = model.CombinedModel(
                 n_feats=n_feats_emb,
                 n_hidden_u=emb_n_hidden_u,
                 n_hidden1_u=discrim_n_hidden1_u,
                 n_hidden2_u=discrim_n_hidden2_u,
                 n_targets=n_targets,
                 param_init=None,
                 eps=1e-4, # Theano uses 1e-4 for batch norm instead of PyTorch default of 1e-5
                 input_dropout=0.,
                 incl_softmax=True) # theano includes softmax in output

    #  feat emb model
    convert_theano_array_to_pytorch_tensor(comb_model.feat_emb.hidden_1.weight, theano_model_params['arr_0'])
    convert_theano_array_to_pytorch_tensor(comb_model.feat_emb.hidden_2.weight, theano_model_params['arr_1'])

    #  embedding
    emb = torch.tensor(theano_model_params['arr_2'])

    #  disc model
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.fat_bias, theano_model_params['arr_3'])

    convert_theano_bn_pytorch(comb_model.disc_net.bn1, 
                          theano_model_params['arr_4'], 
                          theano_model_params['arr_5'], 
                          theano_model_params['arr_6'], 
                          theano_model_params['arr_7'])

    convert_theano_array_to_pytorch_tensor(comb_model.disc_net.hidden_2.weight, theano_model_params['arr_8'])
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.hidden_2.bias, theano_model_params['arr_9'])

    convert_theano_bn_pytorch(comb_model.disc_net.bn2, 
                              theano_model_params['arr_10'], 
                              theano_model_params['arr_11'], 
                              theano_model_params['arr_12'], 
                              theano_model_params['arr_13'])

    convert_theano_array_to_pytorch_tensor(comb_model.disc_net.out.weight, theano_model_params['arr_14'])
    convert_theano_array_to_pytorch_tensor_1d(comb_model.disc_net.out.bias, theano_model_params['arr_15'])

    model_to_return = comb_model.to(device)

    if only_discrim_model:

        emb = emb.to(device)
        #  create disc_net from loaded comb_model
        model_to_return = create_disc_model_multi_gpu(model_to_return, 
                                                      emb, device, 
                                                      eps=1e-4, # Theano uses 1e-4 for batch norm instead of PyTorch default of 1e-5
                                                      incl_softmax=True) # Theano includes softmax in model

    return model_to_return
>>>>>>> wip
