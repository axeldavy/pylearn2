"""
This class implements an MLP followed by a CRF.
"""

import functools
import logging
import numpy as np
import warnings

from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg.MRG_RandomStreams import multinomial as theano_multinomial
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer, MLP
from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX

class MLPCRF(Model):
    def __init__(self, mlp, output_size, connections):
        #TODO

    @wraps(Model.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        self.mlp.set_input_space(space)

    @wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        X, Y = data
        rval = self.mlp.get_layer_monitoring_channels(state_below=X)
        #rval['CRF_misclass'] = ??? Y: truth values, X:inputs
        #rval['CRF_Potentials_norm'] = ...
        return rval

    @wraps(get_monitoring_data_specs)
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

    @wraps(set_batch_size)
    def set_batch_size(self):
        #TODO

    @wraps(get_lr_scalers)
    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()

    @wraps(redo_theano)
    def redo_theano(self):
        #TODO

    def get_potentials(self, inputs):
        #TODO

    def calculate_derivates_energy(self, outputs): # why no give the pairwise current deriv as input ? would allow to use less space
        #TODO: This is an innefficient implementation I think. To improve
        def fill_pairwise_derivative(index, neighboors_indexes, P_pairwise_d_current, outputs):
            def fill_pairwise_derivative_for_batch_index(batch_index_, P_pairwise_d_current_, index_, neighboors_indexes_, outputs_):
                def fill_pairwise_derivative_for_neighboor(neighboor_index__, P_pairwise_d_current__, batch_index__, index__, label_index__, outputs__):
                    label_neighboor__ = outputs__[batch_index__, neighboor_index__]
                    return set_subtensor(P_pairwise_d_current__[batch_index, index, label_index__, neighboor_index__, label_neighboor__], 1)
                return theano.scan(fn=fill_pairwise_derivative_for_neighboor, sequences=[neighboors_indexes_], outputs_info=P_pairwise_d_current_, non_sequences=[batch_index_, index_, outputs_[batch_index_, index_], outputs_])[-1]
            return theano.scan(fn=fill_pairwise_derivative_for_batch_index, sequences=[theano.tensor.arange(outputs.shape[0])], outputs_info=[P_pairwise_d_current], non_sequences=[index, neigboors_indexes, outputs])[-1]

        derivative_pairwise = theano.shared(numpy.zeros(self.P_pairwise_size, config.floatX))
        derivative_pairwise = theano.scan(fn=fill_pairwise_derivative, sequences=[theano.tensor.arange(outputs.shape[1])], outputs_info=[derivative_pairwise], non_sequences=[neigboors_indexes, outputs], n_steps=self.num_indexes)[-1]

        derivative_unaries = theano.shared(numpy.zeros(self.P_unaries_size, config.floatX))
        derivative_unaries = theano.scan(fn=lambda batch_index, derivative_unaries_current, outputs: set_subtensor(derivative_unaries_current[batch_index, :, outputs[batch_index, :]], 1),
                                         sequences=[theano.tensor.arange(outputs.shape[0])], outputs_info=derivative_unaries, non_sequences=[outputs])[-1]
        return derivative_unaries, derivative_pairwise

    def gibbs_sample_step(self, P_unaries, P_pairwise, current_output):
        def update_case(index, neighboors_indexes, current_output, P_unaries, P_pairwise):
            sum_P_pairwise = theano.map(fn=lambda batch_index, index, neigboors_indexes, current_output, P_pairwise: P_pairwise[batch_index, index, :, neigboors_indexes, current_output[batch_index, neigboors_indexes]].sum(axis=1), sequences=[theano.tensor.arange(current_output.shape[0])], non_sequences=[index, neigboors_indexes, current_output, P_pairwise])
            P_for_labels = P_unaries[:, index, :] +  sum_P_pairwise
            probabilities = P_for_labels / P_for_labels.sum(axis=1) # batch_size x num_labels
            update_case = theano_multinomial(probabilities)
            new_output = set_subtensor(current_output[index], update_case)
            return new_output
        Outputs = theano.scan(fn=update_case, sequences=[theano.tensor.arange(current_output.shape[1]), self.connections], outputs_info=[current_output], non_sequences=[P_unaries, P_pairwise], n_steps=self.num_indexes)
        return Outputs[-1]