import numpy as np

import torch
import torch.nn.functional as F


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


def get_attributions(test_generator, set_size, discrim_model, filename='data.h5'):
    
    #  import captum here in case user doesn't have it installed!
    from captum.attr import IntegratedGradients
    import h5py
    
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('integrated_gradients', shape=(set_size, test_generator.dataset.xs.shape[1]))
        # if you wanted to add other attribution methods here:
        #hf.create_dataset('other attribution method name', shape=(set_size, test_generator.dataset.xs.shape[1]))

        #  compute attributions at end of training
        with torch.no_grad():
            #  initialize attribution methods here
            ig = IntegratedGradients(discrim_model)

            idx = 0
            for x_batch, y_batch, _ in test_generator:

                # Compute attribution methods here
                attr, delta = ig.attribute(inputs=(x_batch.cpu()), target=y_batch.cpu(), return_convergence_delta=True, n_steps=50)
                attr = attr.detach().numpy()
                hf['integrated_gradients'][idx:idx+len(x_batch)] = attr
                idx += len(x_batch)
                print('completed {}/{} [{:3f}%]'.format(idx, set_size, 100*idx/set_size))
