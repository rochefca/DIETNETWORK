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

        # For each class, get the integrated gradients of that class' predictions
        # wrt the inputs and then sort by average value of that gradient
        grads_wrt_inputs = []
        interpolation_factors = (torch.tensor((np.arange(n_steps, dtype="float32") / n_steps)[:, None])).to(baselines.device)
        itt_inputs = baselines + interpolation_factors * (inputs - baselines)

        gs = self._grad_from_normed_fn(itt_inputs, target) * (inputs - baselines)

        #  keep dims consistent with captum shapes/sizes 
        #  (including outputting extra object if return_convergence_delta)
        if return_convergence_delta:
            return gs.mean(0)[None,:], None
        else:
            return gs.mean(0)[None,:]

    def _grad_from_normed_fn(self, normed_input_sup, class_idx):
        # Obtain function that takes as inputs normed inputs and returns the
        # gradient of a class score wrt the normed inputs themselves (this is
        # requird because computing the integrated gradients requires to be able
        # to interpolate between an example where all features are missing and an
        # example where any number of features are provided)

        normed_input_sup.requires_grad = True
        prediction_sup_det = self.model(normed_input_sup)
        prediction_sup_det[:, class_idx].sum().backward()
        return normed_input_sup.grad


"""
# keep these here for now!
class_idx = T.iscalar("class index")
grad_fn = theano.function([input_var_sup, class_idx],
                          T.grad(prediction_sup_det[:, class_idx].mean(), input_var_sup).mean(0))
grads_wrt_inputs = mlh.get_grads_wrt_inputs(x_test, grad_fn, feature_names,
                                            label_names)

# Obtain function that takes as inputs normed inputs and returns the
# gradient of a class score wrt the normed inputs themselves (this is
# requird because computing the integrated gradients requires to be able
# to interpolate between an example where all features are missing and an
# example where any number of features are provided)
grad_from_normed_fn = theano.function(
                        [normed_input_sup, class_idx],
                        T.grad(prediction_sup_det[:, class_idx].sum(), normed_input_sup).mean(0))

# Collect integrated gradients over the whole test set. Obtain, for each
# SNP, for each possible value (0, 1 or 2), the average contribution of that
# value for what SNP to the score of each class.
avg_int_grads = np.zeros((x_test.shape[1], 3, len(label_names)), dtype="float32")
counts_int_grads = np.zeros((x_test.shape[1], 3), dtype="int32")
for test_idx in range(x_test.shape[0]):
    int_grads = mlh.get_integrated_gradients(x_test[test_idx], grad_from_normed_fn,
                                             feature_names, label_names, norm_mus,
                                             norm_sigmas, m=100)

    snp_value_mask = np.arange(3) == x_test[test_idx][:, None]
    avg_int_grads += snp_value_mask[:, :, None] * int_grads.transpose()[:, None, :]
    counts_int_grads += snp_value_mask
avg_int_grads = avg_int_grads / counts_int_grads[:, :, None]

def get_grads_wrt_inputs(inputs, grad_fn, input_names, class_names):
    # For each class, get the gradients of that class' prediction wrt the 
    # inputs and then sort by average value of that gradient
    grads_wrt_inputs = {}
    for i in range(len(class_names)):
        gs = grad_fn(inputs, i)
        sorted_gs = sorted(zip(gs, input_names))
        grads_wrt_inputs[class_names[i]] = sorted_gs

    return grads_wrt_inputs


def get_integrated_gradients(inputs, grad_from_normed_fn, input_names, class_names,
                             norm_mus, norm_sigmas, m=100):

    # Normalize inputs and obtain reference inputs
    normed_inputs = ((inputs == -1) * norm_mus +
                     (inputs != -1) * inputs.astype("float32"))
    normed_inputs = (normed_inputs - norm_mus) / norm_sigmas
    ref_inputs = normed_inputs * 0

    # For each class, get the integrated gradients of that class' predictions
    # wrt the inputs and then sort by average value of that gradient
    grads_wrt_inputs = []
    interpolation_factors = (np.arange(m, dtype="float32") / m)[:, None]
    itt_inputs = ref_inputs + interpolation_factors * (normed_inputs - ref_inputs)
    for i in range(len(class_names)):

        # Compute integrated gradients
        gs = grad_from_normed_fn(itt_inputs, i) * (normed_inputs - ref_inputs)
        grads_wrt_inputs.append(gs)

    return np.stack(grads_wrt_inputs)
"""