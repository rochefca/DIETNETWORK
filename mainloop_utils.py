import numpy as np

import torch
import torch.nn.functional as F


def eval_step(valid_generator, set_size, discrim_model, criterion):
    discrim_model.eval()

    valid_minibatch_mean_losses = []
    valid_minibatch_n_right = [] # nb of good classifications

    for x_batch, y_batch, _ in valid_generator:
        # Forward pass
        discrim_model_out = discrim_model(x_batch)

        # Predictions
        pred = get_predictions(discrim_model_out)

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
        yhat = F.softmax(model_output, dim=1)
        _, pred = torch.max(yhat, dim=1)

    return pred


def compute_accuracy(n_right, set_size):
    acc = np.array(n_right).sum() / float(set_size)*100

    return acc


def has_improved(best_acc, actual_acc, min_loss, actual_loss):
    if actual_acc > best_acc:
        return True
    if actual_acc == best_acc and actual_loss < min_loss:
        return True

    return False


def test_step(test_generator, test_size, discrim_model):
    pass
