import torch
import numpy as np

class AbstractAttributionMethod:
    def __init__(self):
        raise NotImplementedError
    def attribute():
        raise NotImplementedError

#TODO: fill in this class based on the implementation of Pierre-Luc
class IntegratedGradients(AbstractAttributionMethod):
    def __init__(self, model, **kwargs):
        self.model = model

    def attribute(self, inputs, target, n_steps, baselines, return_convergence_delta=False, **ignore):
        #  following original, using normalized in input, so no need for normalization params
        #  renamed m to n_steps to match captum variable names
        #  if convergence_delta is passed it will be ignored
        #  must pass device here!

        #  For each class, get the integrated gradients of that class' predictions
        #  wrt the inputs and then sort by average value of that gradient
        n_batch, n_feats = inputs.shape

        grads_wrt_inputs = []
        
        #  n_steps x 1
        interpolation_factors = (torch.tensor((np.arange(n_steps, dtype="float32") / n_steps)[:, None])).to(baselines.device)
        
        #  n_steps x n_batch x 1
        interpolation_factors = interpolation_factors.unsqueeze(1).repeat(1, n_batch, 1)
        
        #  (1 x n_feats) + (n_steps x n_batch x 1) * (1 x n_batch x n_feats)
        itt_inputs = baselines + interpolation_factors * (inputs - baselines).unsqueeze(0)
        
        #  (n_steps, n_batch, n_feats) -> (n_batch*n_steps x n_feats)
        itt_inputs = itt_inputs.view(-1, n_feats)
        
        # (n_batch*n_steps x n_feats)
        gs = self._grad_from_normed_fn(itt_inputs, target)

        # (n_steps, n_batch, n_feats)
        gs = gs.view(n_steps, n_batch, n_feats) * (inputs - baselines).unsqueeze(0)

        #  keep dims consistent with captum shapes/sizes 
        #  (including outputting extra object if return_convergence_delta)
        if return_convergence_delta:
            return gs.mean(0), None
        else:
            return gs.mean(0)

    def _grad_from_normed_fn(self, normed_input_sup, class_idx):
        #  Obtain function that takes as inputs normed inputs and returns the
        #  gradient of a class score wrt the normed inputs themselves (this is
        #  requird because computing the integrated gradients requires to be able
        #  to interpolate between an example where all features are missing and an
        #  example where any number of features are provided)

        normed_input_sup.requires_grad = True
        prediction_sup_det = self.model(normed_input_sup)
        prediction_sup_det[:, class_idx].sum().backward()
        return normed_input_sup.grad


class Saliency(AbstractAttributionMethod):
    def __init__(self, model, **kwargs):
        raise NotImplementedError
