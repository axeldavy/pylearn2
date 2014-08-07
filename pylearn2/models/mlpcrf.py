"""
This class implements an MLP followed by a CRF.
"""

import functools
import logging
import numpy as np
import warnings

from theano.compat.python2x import OrderedDict
from theano import tensor as T
from theano import config
import theano

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer, MLP
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import IndexSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils import safe_zip
from pylearn2.utils.rng import make_theano_rng


def one_hot_theano(t, r=None):
    """
    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.
    
    if r is not given, r is set to max(t) + 1
    """
    if r is None:
        r = T.max(t) + 1
        
    ranges = T.shape_padleft(T.arange(r), t.ndim)
    return T.eq(ranges, T.shape_padright(t, 1))


def get_next_16_multiple(num):
    return ((int(num) + 31) / 16) * 16


class MLPCRF(Model):
    """
    This model is a MLP followed by a CRF for the outputs.
    For now, only 2D is supported. The MLP's output must be a
    Conv2DSpace and match the shape expected by the CRF

    Parameters
    ----------
    mlp : object of class MLP
        The mlp below the CRF.
    output_shape : tuple
        The shape of the 2D output grid of the CRF.
    neighborhood : a tuple of tuple of relative neighbors.
            A neighbor is define by a tuple (x, y) which corresponds to
            the relative position of the neighbor of a given node.
    unaries_pool_shape : tuple
        Tells when getting the unary features, which region of
        the MLP outputs to take. For example if set to (3, 3),
        the unary feature would be of length 3x3x(number of the
        output of the MLP).
    num_labels : integer
        The number of labels of the outputs.
    """
    def __init__(self, mlp, output_shape, neighborhood, unaries_pool_shape, num_labels):
        super(MLPCRF, self).__init__()

        if not(isinstance(mlp, MLP)):
            raise ValueError("MLPCRF expects an object of class MLP as input")

        self.mlp = mlp
        self.batch_size = mlp.batch_size
        self.force_batch_size = self.batch_size
        self.output_shape = output_shape
        self.num_indexes = output_shape[0] * output_shape[1]

        self.neighborhood = neighborhood

        self.unaries_pool_shape = unaries_pool_shape
        self.num_labels = num_labels
        self.theano_rng = make_theano_rng(None, 2014+8+7, which_method="multinomial")

        space = self.mlp.get_input_space()
        self.input_space = space

        self.mlp_output_space = self.mlp.get_output_space()
        if not (isinstance(self.mlp_output_space, Conv2DSpace)):
            raise ValueError("MLPCRF expects the MLP to output a Conv2DSpace")

        if self.mlp_output_space.shape[0] <> self.unaries_pool_shape[0] + self.output_shape[0] or\
           self.mlp_output_space.shape[1] <> self.unaries_pool_shape[1] + self.output_shape[1]:
               raise ValueError("MLPCRF expects the MLP output to be of shape [" +\
                                str(self.unaries_pool_shape[0] + self.output_shape[0]) + ", " +\
                                str(self.unaries_pool_shape[1] + self.output_shape[1]) + "] but got " +\
                                str(self.mlp_output_space.shape))

        self.desired_mlp_output_space = Conv2DSpace(shape=self.mlp_output_space.shape,
                                              axes=('c', 0, 1, 'b'),
                                              num_channels=self.mlp_output_space.num_channels)
        self.unaries_convolution = MaxoutConvC01B(
                                                  get_next_16_multiple(num_labels),
                                                  1,
                                                  unaries_pool_shape,
                                                  [1, 1],
                                                  [1, 1],
                                                  'unaries_convolution',
                                                  irange=.005,
                                                  pad=0,
                                                  min_zero=False,
                                                  max_kernel_norm=0.9,
                                                  no_bias=True
                                                  )
        self.pairwise_convolution = MaxoutConvC01B(
                                                  get_next_16_multiple(num_labels ** 2),
                                                  1,
                                                  [1, 1],
                                                  [1, 1],
                                                  [1, 1],
                                                  'pairwise_convolution',
                                                  irange=.005,
                                                  pad=0,
                                                  min_zero=False,
                                                  max_kernel_norm=0.9,
                                                  no_bias=True
                                                  )
        self.unaries_convolution.mlp = self.mlp
        self.pairwise_convolution.mlp = self.mlp
        self.unaries_convolution.set_input_space(self.desired_mlp_output_space)
        self.pairwise_convolution.set_input_space(self.desired_mlp_output_space) #Perhaps something to do here
        self.zeros_output_shape = sharedX(np.zeros((num_labels,) + tuple(self.output_shape) + (self.batch_size,)))

        self.output_space = IndexSpace(num_labels, self.num_indexes)

    @wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        X, Y = data
        rval = self.mlp.get_layer_monitoring_channels(state_below=X)
        #rval['CRF_misclass'] = ??? Y: truth values, X:inputs
        #rval['CRF_Potentials_norm'] = ...
        return rval

    @wraps(Model.get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        """
        Notes
        -----
        In this case, we want the inputs and targets.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    @wraps(Model.get_params)
    def get_params(self):
        params = self.mlp.get_params()
        params_unaries = self.unaries_convolution.get_params()
        for param in params_unaries:
            assert not param in params
            params.append(param)
        params_pairwise = self.pairwise_convolution.get_params()
        for param in params_pairwise:
            assert not param in params
            params.append(param)
        return params

    @wraps(Model.set_batch_size)
    def set_batch_size(self, batch_size):
        self.mlp.set_batch_size(batch_size)
        self.batch_size = batch_size

    @wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()

    def get_potentials(self, inputs):
        """
        Calculate the potentials given a batch of inputs

        Parameters
        ----------
        inputs : member of self.input_space

        Returns
        -------
        P_unaries : (num_labels, rows, cols, batch_size) tensor
            The unary potentials of the CRF.
        P_pairwise : (num_neighbors, num_labels ** 2, rows, cols, batch_size) tensor
            The pairwise potentials of the CRF.
        """
        self.input_space.validate(inputs)

        mlp_outputs_old_space = self.mlp.fprop(inputs)

        mlp_outputs_new_space = self.mlp_output_space.format_as(mlp_outputs_old_space, self.desired_mlp_output_space)

        P_unaries = self.unaries_convolution.fprop(mlp_outputs_new_space)[:self.num_labels, :, :, :]

        zeros = sharedX(np.zeros((self.mlp_output_space.num_channels,) + tuple(self.output_shape) + (self.batch_size,)))
        pairwise_inputs = mlp_outputs_new_space[:,
                                                self.unaries_pool_shape[0]//2:(self.unaries_pool_shape[0]//2 + self.output_shape[0]),
                                                self.unaries_pool_shape[1]//2:(self.unaries_pool_shape[1]//2 + self.output_shape[1]),
                                                :]

        P_pairwise = []


        for (delta_x, delta_y) in self.neighborhood:
            if delta_x > 0:
                slice_x_left = slice(None, -delta_x)
                slice_x_right = slice(delta_x, None)
            elif delta_x < 0:
                slice_x_left = slice(-delta_x, None)
                slice_x_right = slice(None, delta_x)
            else:
                slice_x_left = slice(None)
                slice_x_right = slice(None)

            if delta_y > 0:
                slice_y_left = slice(None, -delta_y)
                slice_y_right = slice(delta_y, None)
            elif delta_y < 0:
                slice_y_left = slice(-delta_y, None)
                slice_y_right = slice(None, delta_y)
            else:
                slice_y_left = slice(None)
                slice_y_right = slice(None)

            input_for_edge = T.set_subtensor(zeros[:, slice_x_left, slice_y_left, :],
                                             pairwise_inputs[:, slice_x_left, slice_y_left, :] - pairwise_inputs[:, slice_x_right, slice_y_right, :])
            input_for_edge = T.abs_(input_for_edge)

            P_pairwise.append(
                self.pairwise_convolution.fprop(input_for_edge)[:self.num_labels ** 2, :, :, :]
                )
        P_pairwise = T.stacklists(P_pairwise)

        return P_unaries, P_pairwise
            

    def calculate_energy(self, P_unaries, P_pairwise, outputs):
        """
        Calculate the energy

        Parameters
        ----------
        P_unaries : (num_labels, rows, cols, batch_size) tensor
            The unary potentials of the CRF.
        P_pairwise : (num_neighbors, num_labels ** 2, rows, cols, batch_size) tensor
            The pairwise potentials of the CRF.
        outputs : (rows, cols, batch_size) tensor

        Returns
        -------
        energy : tensor
        """

        one_hot_output = one_hot_theano(outputs, r=self.num_labels)

        energy_unaries = T.dot(one_hot_output.flatten(), P_unaries.flatten())

        energy_pairwise = 0

        for ((delta_x, delta_y), i) in safe_zip(self.neighborhood, xrange(len(self.neighborhood))):
            if delta_x > 0:
                slice_x_left = slice(None, -delta_x)
                slice_x_right = slice(delta_x, None)
            elif delta_x < 0:
                slice_x_left = slice(-delta_x, None)
                slice_x_right = slice(None, delta_x)
            else:
                slice_x_left = slice(None)
                slice_x_right = slice(None)

            if delta_y > 0:
                slice_y_left = slice(None, -delta_y)
                slice_y_right = slice(delta_y, None)
            elif delta_y < 0:
                slice_y_left = slice(-delta_y, None)
                slice_y_right = slice(None, delta_y)
            else:
                slice_y_left = slice(None)
                slice_y_right = slice(None)

            label_neighbor_combination = T.set_subtensor(self.zeros_output_shape, outputs[slice_x_left, slice_y_left, :] * 5 + outputs[slice_x_right, slice_y_right, :])
            energy_pairwise = energy_pairwise + T.dot(one_hot_theano(label_neighbor_combination, r=(self.num_labels ** 2)).flatten(), P_pairwise[i, ...].flatten())

        return (energy_unaries + energy_pairwise) / self.batch_size, OrderedDict()

    # def gibbs_sample_step(self, P_unaries, P_pairwise, current_outputs):
    #     """
    #     Does one iteration of gibbs sampling.

    #     Parameters
    #     ----------
    #     P_unaries : (num_labels, rows, cols, batch_size) tensor
    #         The unary potentials of the CRF.
    #     P_pairwise : (num_neighbors, num_labels ** 2, rows, cols, batch_size) tensor
    #         The pairwise potentials of the CRF.
    #     current_outputs : (rows, cols, batch_size) tensor

    #     Returns
    #     -------
    #     new_output : (rows, cols, batch_size) tensor
    #     updates : subclass of dictionary specifying the update rules for all shared variables
    #     """

    #     if not hasattr(self, neighbor_theano):
    #         grid_x = np.arange(self.output_shape[0])
    #         grid_y = np.arange(self.output_shape[1])
    #         grid_xy = np.meshgrid(x, y)
    #         self.sequence_x = theano.shared(grid_xy[0].flatten())
    #         self.sequence_y = theano.shared(grid_xy[0].flatten())
    #         self.neighbor_theano = theano.shared(self.neighbor)
    #         self.theano_init_zero = sharedX(0.)

    #     def update_case(x, y, current_outputs, P_unaries):
    #         P_for_labels = T.exp(T.neg(P_unaries[:, x, y, :].T))# +  sum_P_pairwise))
    #         probabilities = P_for_labels / T.sum(P_for_labels, axis=1)[:,None] # num_batches x num_labels
    #         update_case = self.theano_rng.multinomial(pvals=probabilities)
    #         update_case = T.argmax(update_case, axis=1) # convert from one_hot
    #         new_output = T.set_subtensor(current_outputs[:, index], update_case)
    #         return new_output#, update
    #     scan_outputs, scan_updates = theano.scan(fn=update_case,
    #                                              sequences=[self.sequence_x,
    #                                                         self.sequence_y],
    #                                              outputs_info=[current_outputs],
    #                                              non_sequences=[P_unaries])
    #     return scan_outputs[-1], scan_updates
