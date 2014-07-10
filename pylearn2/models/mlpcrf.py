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
    """
    This model is a MLP followed by a CRF for the outputs.
    For now, only 2D is supported. The MLP's output must be a
    Conv2DSpace and match the shape expected by the CRF

    Parameters
    ----------
    mlp : object of class MLP
        The mlp below the CRF.
    output_size : tuple
        The shape of the 2D output grid of the CRF.
    connections : list of list [TO CHANGE]
        Describes the connections to other indexes
    unaries_pool_shape : tuple
        Tells when getting the unary features, which region of
        the MLP outputs to take. For example if set to (3, 3),
        the unary feature would be of length 3x3x(number of the
        output of the MLP).
    num_labels : integer
        The number of labels of the outputs.
    """
    def __init__(self, mlp, output_size, connections, unaries_pool_shape, num_labels):
        super(MLPCRF, self).__init__()

        if not(isinstance(mlp, MLP)):
            raise ValueError("MLPCRF expects an object of class MLP as input")
        self.output_size = output_size
        self.num_indexes = output_size[0] * output_size[1]
        self.connections = connections ???
        self.unaries_pool_shape = unaries_pool_shape ???
        self.num_labels = num_labels

    @wraps(Model.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        self.mlp.set_input_space(space)

        self.mlp_output_space = self.mlp.get_output_space()
        if not (isinstance(self.mlp_output_space, Conv2DSpace)):
            raise ValueError("MLPCRF expects the MLP to output a Conv2DSpace")

        if self.mlp_output_space.shape[0] <> self.unaries_pool_shape[0] + self.output_size[0] or
           self.mlp_output_space.shape[1] <> self.unaries_pool_shape[1] + self.output_size[1]:
               raise ValueError("MLPCRF expects the MLP output to be of shape [" +\
                                str(self.unaries_pool_shape[0] + self.output_size[0]) + ", " +\
                                str(self.unaries_pool_shape[1] + self.output_size[1]) + "] but got " +\
                                str(self.mlp_output_space.shape))

        self.desired_mlp_output_space = Conv2DSpace(shape=self.unaries_pool_shape,
                                              axes=('b', 0, 1, 'c'),
                                              num_channels=self.mlp_output_space.num_channels)

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
    def set_batch_size(self, batch_size):
        self.mlp.set_batch_size(batch_size)
        self.batch_size = batch_size

    @wraps(get_lr_scalers)
    def get_lr_scalers(self):
        return self.mlp.get_lr_scalers()

    """
    Current Big problem about the implementation:
    conceptually we want the connections with the neighboors
    to be a list of list. However this doesn't exist in Theano,
    and you have to have a matrix.
    I do not handle that right.
    One idea was that additional cases should point to an added case
    of every tensor with one axis beeing indexes, but that seems very bad.
    But now I think we should introduce a Vector telling how many neighboors
    there are for every index, and use that.
    """
    def get_potentials(self, inputs):
        """
        Calculate the potentials given a batch of inputs

        Parameters
        ----------
        inputs : member of self.input_space

        Returns
        -------
        P_unaries : (num_batches, num_indexes, num_labels) tensor
            The unary potentials of the CRF.
        P_pairwise : (num_batches, num_indexes, num_labels, num_indexes, num_labels) tensor
            The pairwise potentials of the CRF.
            the fourth and fifth dim are for the neighboor of the point we consider
            (according to the defined connections).
            The content is undefined for indexes for non-neighboors
        """
        self.input_space.validate(inputs)

        mlp_outputs_old_space = self.mlp.fprop(inputs)

        mlp_outputs_new_space = self.mlp_output_space.format_as(mlp_outputs_old_space, self.desired_mlp_output_space)
        P_unaries = T.TensorType(config.floatX , (False,)*3)()
        P_unaries = T.specify_shape(P_unaries, (self.batch_size, self.num_indexes, self.num_labels))
        P_pairwise = T.TensorType(config.floatX , (False,)*5)()
        P_pairwise = T.specify_shape(P_pairwise, (self.batch_size, self.num_indexes, self.num_labels, self.num_indexes, self.num_labels))

        """
        Fill the unary potentials.
        Does an equivalent of:
        for i in indexes:
            u = vectors of outputs seen by the CRF node from the MLP across the batch
            for l in labels:
                for b in batch:
                    P_unaries[b, i, l] = scalar_product(W[l], u[b])
        
        """

        def fill_unaries_for_index(???, index, P_unaries_current, mlp_outputs, unaries_vectors):
            def compute_scalar_for_label(label, P_unaries_current_, index_, mlp_outputs_seen_, unaries_vectors_):
                return set_subtensor(P_unaries_current_[:, index_, label], T.prod(unaries_vectors_[label, :], mlp_outputs_seen_))
            
            mlp_outputs_seen = mlp_outputs[:, ???, ???, :].reshape(mlp_outputs.shape[0], -1)
            return theano.scan(fn=compute_scalar_for_label, sequences=[T.arange(self.num_labels)], outputs_info=[P_unaries_current], non_sequences=[index, mlp_outputs_seen, unaries_vectors_])[-1]
        P_unaries = theano.scan(fn=fill_unaries_for_index, sequences=[??, T.arange(self.num_indexes)], outputs_info=[P_unaries], non_sequences=[mlp_outputs_new_space, self.unaries_vectors])[-1]

        """
        Fill the pairwise potentials.
        Does an equivalent of:
        for i in indexes:
            u1 = vectors for i across the batch
            for v in neighboors(i):
                u2 = vectors for v across the batch
                for li in labels:
                    for lv in labels:
                        P_pairwise[:, i, li, v, lv] = scalar_product(W'[li, lv], |u1-u2|)
        """

        def fill_pairwise_for_label_neighboor_i4(label_neighboor, P_pairwise_current, index, index_neighboor, label_index, feature_index, feature_neigboor, pairwise_vectors):
            potential = T.prod(pairwise_vectors[label_index, label_neighboor], T.abs_(feature_index - feature_neighbor))
            return set_subtensor(P_pairwise_current[:, index, label_index, index_neighboor, label_neighboor], potential)

        for fill_pairwise_for_label_index_i3(label_index, P_pairwise_current, index, index_neighboor, feature_index, feature_neigboor, pairwise_vectors):
            return theano.scan(fn=fill_pairwise_for_label_neighboor_i4 , sequences=[T.arange(self.num_labels)], outputs_info=[P_unaries_current], non_sequences=[index, index_neighboor, label_index, feature_index, feature_neigboor, pairwise_vectors])[-1]

        def fill_pairwise_for_index_and_neighboor_i2(index_neighboor, P_pairwise_current, index, feature_index, mlp_outputs, pairwise_vectors):
            feature_neigboor = mlp_outputs[:, ?, ?, :]
            return theano.scan(fn=fill_pairwise_for_label_index_i3, sequences=[T.arange(self.num_labels)], outputs_info=[P_unaries_current], non_sequences=[index, index_neighboor, feature_index, feature_neigboor, pairwise_vectors])[-1]

        def fill_pairwise_for_index_i1(index, neighboors, P_pairwise_current, mlp_outputs, pairwise_vectors):
            feature_index = mlp_outputs[:, ?, ?, :]
            return theano.scan(fn=fill_pairwise_for_index_and_neighboor_i2, sequences=[neighboors], outputs_info[P_pairwise_current], non_sequences=[index, feature_index, mlp_outputs, pairwise_vectors])[-1]
        
        P_pairwise = theano.scan(fn=fill_pairwise_for_index_i1, sequences=[T.arange(self.num_labels), self.connections], outputs_info=[P_pairwise], non_sequences=[mlp_outputs, self.pairwise_vectors])[-1]

        return P_unaries, P_pairwise



    def calculate_derivates_energy(self, outputs, d_unaries_to_update=None, d_pairwise_to_update=None):
        """
        Calculate the derivatives of the energy given
        the unary and pairwise potentials

        Parameters
        ----------
        outputs : (num_batches, num_indexes) tensor

        Returns
        -------
        derivative_unaries : (num_batches, num_indexes, num_labels) tensor
            The derivative given theunary potentials of the CRF.
        derivative_pairwise : (num_batches, num_indexes, num_labels, num_indexes, num_labels) tensor
            The derivative given The pairwise potentials of the CRF.
        """

        if d_unaries_to_update is None:
            derivative_unaries = theano.shared(numpy.zeros((self.batch_size, self.num_indexes, self.num_labels), config.floatX))
        else:
            derivative_unaries = d_unaries_to_update

        if d_pairwise_to_update is None:
            derivative_pairwise = theano.shared(numpy.zeros((self.batch_size, self.num_indexes, self.num_labels, self.num_indexes, self.num_labels), config.floatX))
        else:
            derivative_pairwise = d_pairwise_to_update

        """
        Inefficient I think.
        Fill the pairwise potentials derivatives.
        Does an equivalent of:
        for i in index:
            for b in batches;
               li = outputs[b, i]
               for v in neighboors(i):
                   lv = outputs[b, v]
                   derivative[b, i, li, v, lv] += 1
        """
        def fill_pairwise_derivative(index, neighboors_indexes, P_pairwise_d_current, outputs):
            def fill_pairwise_derivative_for_batch_index(batch_index_, P_pairwise_d_current_, index_, neighboors_indexes_, outputs_):
                def fill_pairwise_derivative_for_neighboor(neighboor_index__, P_pairwise_d_current__, batch_index__, index__, label_index__, outputs__):
                    label_neighboor__ = outputs__[batch_index__, neighboor_index__]
                    return set_subtensor(P_pairwise_d_current__[batch_index, index, label_index__, neighboor_index__, label_neighboor__], P_pairwise_d_current__[batch_index, index, label_index__, neighboor_index__, label_neighboor__] + 1)
                return theano.scan(fn=fill_pairwise_derivative_for_neighboor, sequences=[neighboors_indexes_], outputs_info=P_pairwise_d_current_, non_sequences=[batch_index_, index_, outputs_[batch_index_, index_], outputs_])[-1]
            return theano.scan(fn=fill_pairwise_derivative_for_batch_index, sequences=[theano.tensor.arange(outputs.shape[0])], outputs_info=[P_pairwise_d_current], non_sequences=[index, neigboors_indexes, outputs])[-1]

        derivative_pairwise = theano.scan(fn=fill_pairwise_derivative, sequences=[theano.tensor.arange(outputs.shape[1])], outputs_info=[derivative_pairwise], non_sequences=[neigboors_indexes, outputs], n_steps=self.num_indexes)[-1]

        """
        Fill the unary potentials derivatives.
        Does an equivalent of:
        for b in batches:
            derivative[b, :, outputs[b, :]] += 1
        """

        
        derivative_unaries = theano.scan(fn=lambda batch_index, derivative_unaries_current, outputs: set_subtensor(derivative_unaries_current[batch_index, :, outputs[batch_index, :]], derivative_unaries_current[batch_index, :, outputs[batch_index, :]] + 1),
                                         sequences=[theano.tensor.arange(outputs.shape[0])], outputs_info=derivative_unaries, non_sequences=[outputs])[-1]
        return derivative_unaries, derivative_pairwise

    def gibbs_sample_step(self, P_unaries, P_pairwise, current_output):
        """
        Does one iteration of gibbs sampling.

        Parameters
        ----------
        P_unaries : (num_batches, num_indexes, num_labels) tensor
            The unary potentials of the CRF.
        P_pairwise : (num_batches, num_indexes, num_labels, num_indexes, num_labels) tensor
            The pairwise potentials of the CRF.
        current_output : (num_batches, num_indexes) tensor

        Returns
        -------
        new_output : (num_batches, num_indexes) tensor
        """
        """
        Does one iteration of gibbs sampling.
        Does an equivalent of:
        for i in index: #Would be better if index order is random
            # does one update step
            for b in batches:
                sum_P_pairwise_for_i[b] = P_pairwise[b, i, :, list_of_neighboors, outputs[list_of_neighboors]].sum(axis=1)
        """
        def update_case(index, neighboors_indexes, current_output, P_unaries, P_pairwise):
            sum_P_pairwise = theano.map(fn=lambda batch_index, index, neigboors_indexes, current_output, P_pairwise: P_pairwise[batch_index, index, :, neigboors_indexes, current_output[batch_index, neigboors_indexes]].sum(axis=1), sequences=[theano.tensor.arange(current_output.shape[0])], non_sequences=[index, neigboors_indexes, current_output, P_pairwise])
            P_for_labels = T.exp(T.neg(P_unaries[:, index, :] +  sum_P_pairwise))
            probabilities = P_for_labels / P_for_labels.sum(axis=1) # num_batches x num_labels
            update_case = theano_multinomial(probabilities) # num_batches
            new_output = set_subtensor(current_output[index], update_case)
            return new_output
        Outputs = theano.scan(fn=update_case, sequences=[theano.tensor.arange(current_output.shape[1]), self.connections], outputs_info=[current_output], non_sequences=[P_unaries, P_pairwise], n_steps=self.num_indexes)
        return Outputs[-1]