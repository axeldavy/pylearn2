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
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import IndexSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils.rng import make_theano_rng

class CRFNeighborhood():
    """
    Implements the definition of a neighborhood for a CRF.
    A CRFNeighborhood is initialized according to the size
    of a rectangular lattice and a tuple of tuples which indicate
    the relative position of the neighbors of the current_node.


    Parameters
    ----------
    neighbors : a 2D matrix (theano tensor). Each raw i contains
        the neighbors of the node i, plus eventually some meaningless 0.
    neighbors_sizes : a vector which contains the sizes of the neighborhood
        for each node in the graph. Using it help to stop before the
        meaningless 0 of neighbors. 
    """

    def __init__(self, lattice_size, neighborhood_shape):
        """
        Creates an instance of the class given the size of a rectangular lattice
        and a neighborhood shape. The nodes are indexed from left to right and
        from top to bottom.

        Parameters
        ----------
        lattice_size : a 2D vector which contains the size of the lattice.
            The first elements is the number of raws in the lattice, the second
            is for the columns.
        neighborhood_shape : a tuple of tuple of relative neighbors.
            A neighbor is define by a tuple (x, y) which corresponds to
            the relative position of the neighbor of a given node.
        """
        neighborhoods_dict = dict()
        # Iterate over the lattice
        for y_current in range(lattice_size[0]):
            for x_current in range(lattice_size[1]):
                # Creates a list of neighbors if they are in the lattice
                current_neighborhood = []
                for current_neighbor in neighborhood_shape:
                    validate_neighbor = True

                    if (x_current + current_neighbor[0]) < 0:
                        validate_neighbor = False
                    if (x_current + current_neighbor[0]) >= lattice_size[0]:
                        validate_neighbor = False

                    if (y_current + current_neighbor[1]) < 0:
                        validate_neighbor = False
                    if (y_current + current_neighbor[1]) >= lattice_size[1]:
                        validate_neighbor = False

                    if validate_neighbor:
                        current_neighborhood.append((y_current+current_neighbor[1])*lattice_size[1] + x_current+current_neighbor[0])

                neighborhoods_dict[y_current*lattice_size[1] + x_current] = current_neighborhood
        # Changes the type of the dictionnary into theano tensors
        self.neighbors_to_theano_tensor(lattice_size, neighborhoods_dict)

    def neighbors_to_theano_tensor(self, lattice_size, neighborhoods_dict):
        """
        Creates two theano tensors which will contain the neighborhoods
        and the sizes of these neighborhoods.

        Parameters
        ----------
        lattice_size : 2D vector which contains the size of the lattice.
        neighborhoods_dict : A python dictionnaire which contains the
            indexes of the neighbors of the nodes in the graph.
        """
        lattice_length = lattice_size[0]*lattice_size[1]
        self.neighborhoods_sizes = np.zeros((lattice_length)).astype(int)
        for current_node in range(lattice_length):
            self.neighborhoods_sizes[current_node] = len(neighborhoods_dict[current_node])

        self.neighborhoods = np.zeros((lattice_length, np.max(self.neighborhoods_sizes))).astype(int)
        for current_node in range(lattice_length):
            self.neighborhoods[current_node, 0:self.neighborhoods_sizes[current_node]] = neighborhoods_dict[current_node]

        self.neighborhoods_sizes = theano.shared(self.neighborhoods_sizes)
        self.neighborhoods = theano.shared(self.neighborhoods)

def get_window_bounds_for_index(output_size, unaries_pool_shape):
    """
    TODO
    """
    index = 0
    bounds = np.zeros((output_size[0] * output_size[1], 4), np.int)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            bounds[index, 0] = i
            bounds[index, 1] = i + unaries_pool_shape[0]
            bounds[index, 2] = j
            bounds[index, 3] = j + unaries_pool_shape[1]
            index += 1
    return theano.shared(bounds)

def get_window_center_for_index(output_size, unaries_pool_shape):
    """
    TODO
    """
    index = 0
    centers = np.zeros((output_size[0] * output_size[1], 2), np.int)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            centers[index, 0] = i + (unaries_pool_shape[0])//2
            centers[index, 1] = j + (unaries_pool_shape[1])//2
            index += 1
    return theano.shared(centers)


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
    crf_neighborhood : CRFNeighborhood object
        Describes connections to other indexes
    unaries_pool_shape : tuple
        Tells when getting the unary features, which region of
        the MLP outputs to take. For example if set to (3, 3),
        the unary feature would be of length 3x3x(number of the
        output of the MLP).
    num_labels : integer
        The number of labels of the outputs.
    """
    def __init__(self, mlp, output_size, crf_neighborhood, unaries_pool_shape, num_labels):
        super(MLPCRF, self).__init__()

        if not(isinstance(mlp, MLP)):
            raise ValueError("MLPCRF expects an object of class MLP as input")
        if not (isinstance(crf_neighborhood, CRFNeighborhood)):
            raise ValueError("MLPCRF expects an object of class CRFNeighborhood as input")
        self.mlp = mlp
        self.batch_size = mlp.batch_size
        self.force_batch_size = self.batch_size
        self.output_size = output_size
        self.num_indexes = output_size[0] * output_size[1]
        self.neighbors = crf_neighborhood.neighborhoods
        self.neighborhoods_sizes = crf_neighborhood.neighborhoods_sizes
        self.unaries_pool_shape = unaries_pool_shape
        self.window_bounds_for_index = get_window_bounds_for_index(output_size, unaries_pool_shape)
        self.window_centers = get_window_center_for_index(output_size, unaries_pool_shape)
        self.num_labels = num_labels
        self.theano_rng = make_theano_rng(None, 2014+8+7, which_method="multinomial")

        space = self.mlp.get_input_space()
        self.input_space = space

        self.mlp_output_space = self.mlp.get_output_space()
        if not (isinstance(self.mlp_output_space, Conv2DSpace)):
            raise ValueError("MLPCRF expects the MLP to output a Conv2DSpace")

        if self.mlp_output_space.shape[0] <> self.unaries_pool_shape[0] + self.output_size[0] or\
           self.mlp_output_space.shape[1] <> self.unaries_pool_shape[1] + self.output_size[1]:
               raise ValueError("MLPCRF expects the MLP output to be of shape [" +\
                                str(self.unaries_pool_shape[0] + self.output_size[0]) + ", " +\
                                str(self.unaries_pool_shape[1] + self.output_size[1]) + "] but got " +\
                                str(self.mlp_output_space.shape))

        self.desired_mlp_output_space = Conv2DSpace(shape=self.mlp_output_space.shape,
                                              axes=('b', 0, 1, 'c'),
                                              num_channels=self.mlp_output_space.num_channels)
        self.pairwise_vectors = sharedX(np.zeros((num_labels, num_labels, self.mlp_output_space.num_channels)))
        self.unaries_vectors = sharedX(np.zeros((unaries_pool_shape[0] * unaries_pool_shape[1] * self.mlp_output_space.num_channels, num_labels)))
        self.output_space = IndexSpace(num_labels, output_size[0] * output_size[1])

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
        params.append(self.unaries_vectors)
        params.append(self.pairwise_vectors)
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
        P_unaries : (num_batches, num_indexes, num_labels) tensor
            The unary potentials of the CRF.
        P_pairwise : (num_batches, num_indexes, num_labels, num_indexes, num_labels) tensor
            The pairwise potentials of the CRF.
            the fourth and fifth dim are for the neighbor of the point we consider
            (according to the defined connections).
            The content is undefined for indexes for non-neighbors
        updates : subclass of dictionary specifying the update rules for all shared variables
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
            P_unaries[:, i, :] = u * W^T # W: label x num_channels
        
        """

        def fill_unaries_for_index(bounds, index, P_unaries_current, mlp_outputs, unaries_vectors):            
            mlp_outputs_seen = mlp_outputs[:, bounds[0]:bounds[1], bounds[2]:bounds[3], :].reshape(mlp_outputs.shape[0], -1)
            return T.set_subtensor(P_unaries_current[:, index, :], T.dot(mlp_outputs_seen, unaries_vectors.T))
        scan_outputs, scan_updates_unaries = theano.scan(fn=fill_unaries_for_index, sequences=[self.window_bounds_for_index, T.arange(self.num_indexes)], outputs_info=[P_unaries], non_sequences=[mlp_outputs_new_space, self.unaries_vectors])
        P_unaries = scan_outputs[-1]

        """
        Fill the pairwise potentials.
        Does an equivalent of:
        for i in indexes:
            u1 = vectors for i across the batch
            for v in neighbors(i):
                u2 = vectors for v across the batch
                for li in labels:
                    for lv in labels:
                        P_pairwise[:, i, li, v, lv] = |u1-u2| * W'[li, lv] # to optimise with tensordot
        """

        def fill_pairwise_for_label_neighbor_i4(label_neighbor, P_pairwise_current, index, index_neighbor, label_index, feature_index, feature_neighbor, pairwise_vectors):
            potential = T.dot(T.abs_(feature_index - feature_neighbor), pairwise_vectors[label_index, label_neighbor])
            return T.set_subtensor(P_pairwise_current[:, index, label_index, index_neighbor, label_neighbor], potential)

        def fill_pairwise_for_label_index_i3(label_index, P_pairwise_current, index, index_neighbor, feature_index, feature_neighbor, pairwise_vectors):
            scan_outputs, scan_updates = theano.scan(fn=fill_pairwise_for_label_neighbor_i4 , sequences=[T.arange(self.num_labels)], outputs_info=[P_pairwise_current], non_sequences=[index, index_neighbor, label_index, feature_index, feature_neighbor, pairwise_vectors])
            return scan_outputs[-1], scan_updates

        def fill_pairwise_for_index_and_neighbor_i2(index_neighbor, P_pairwise_current, index, feature_index, mlp_outputs, pairwise_vectors):
            feature_neighbor = mlp_outputs[:, self.window_centers[index_neighbor, 0], self.window_centers[index_neighbor, 1], :]
            scan_outputs, scan_updates = theano.scan(fn=fill_pairwise_for_label_index_i3, sequences=[T.arange(self.num_labels)], outputs_info=[P_pairwise_current], non_sequences=[index, index_neighbor, feature_index, feature_neighbor, pairwise_vectors])
            return scan_outputs[-1], scan_updates

        def fill_pairwise_for_index_i1(index, location, neighbors, neighborhoods_size, P_pairwise_current, mlp_outputs, pairwise_vectors):
            feature_index = mlp_outputs[:, location[0], location[1], :]
            scan_outputs, scan_updates = theano.scan(fn=fill_pairwise_for_index_and_neighbor_i2, sequences=[neighbors[:neighborhoods_size]], outputs_info=[P_pairwise_current], non_sequences=[index, feature_index, mlp_outputs, pairwise_vectors])
            return scan_outputs[-1], scan_updates
        
        scan_outputs, scan_updates_pairwise = theano.scan(fn=fill_pairwise_for_index_i1, sequences=[T.arange(self.num_labels), self.window_centers, self.neighbors, self.neighborhoods_sizes], outputs_info=[P_pairwise], non_sequences=[mlp_outputs_new_space, self.pairwise_vectors])
        P_pairwise = scan_outputs[-1]

        scan_updates_unaries.update(scan_updates_pairwise)

        return P_unaries, P_pairwise, scan_updates_unaries

    def calculate_energy(self, P_unaries, P_pairwise, outputs):
        """
        Calculate the energy

        Parameters
        ----------
        P_unaries : (num_batches, num_indexes, num_labels) tensor
            The unary potentials of the CRF.
        P_pairwise : (num_batches, num_indexes, num_labels, num_indexes, num_labels) tensor
            The pairwise potentials of the CRF.
        outputs : (num_batches, num_indexes) tensor

        Returns
        -------
        energy : (num_batches) tensor
        """
        """
        Calculate the pairwise potentials energy.
        Does an equivalent of:
        for b in batches
            for i in index:
                li = outputs[b, i]
                lv = outputs[b, neighbors(i)]
                energy[b] += P_pairwise[b, i, li, v, lv].sum()
        """
        def fill_pairwise_energy_for_index(index, neighbors_index, neighborhoods_size, current_energy, batch):
            label_index = outputs[batch, index]
            label_neighbor = outputs[batch, neighbors_index[:neighborhoods_size]]
            return current_energy + P_pairwise[batch, index, label_index, neighbors_index[:neighborhoods_size], label_neighbor].sum()

        def fill_pairwise_energy_for_batch(batch):
            scan_outputs, scan_updates = theano.scan(fn=fill_pairwise_energy_for_index, sequences=[theano.tensor.arange(self.num_indexes), self.neighbors, self.neighborhoods_sizes], outputs_info=[sharedX(0)], non_sequences=[batch])
            return scan_outputs[-1], scan_updates

        scan_outputs, scan_updates = theano.map(fn=fill_pairwise_energy_for_batch, sequences=[theano.tensor.arange(outputs.shape[0])])
        energy_pairwise = scan_outputs

        def fill_unary_energy_for_index(index, current_energy, batch):
            label = outputs[batch, index]
            return current_energy + P_unaries[b, index, label]

        def fill_unary_energy_for_batch(batch):
            return theano.reduce(fn=fill_unary_energy_for_index, sequences=[theano.tensor.arange(self.num_indexes)], outputs_info=[sharedX(0)], non_sequences=[batch])

        energy_unaries, unaries_update = theano.map(fn=fill_unaries_for_batch, sequences=[theano.tensor.arange(outputs.shape[0])])
        scan_updates.update(unaries_update)
        return energy_unaries + energy_pairwise, scan_updates
            

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
            The derivative given the unary potentials of the CRF.
        derivative_pairwise : (num_batches, num_indexes, num_labels, num_indexes, num_labels) tensor
            The derivative given The pairwise potentials of the CRF.
        updates : subclass of dictionary specifying the update rules for all shared variables
        """

        if d_unaries_to_update is None:
            derivative_unaries = sharedX(np.zeros((self.batch_size, self.num_indexes, self.num_labels), config.floatX))
        else:
            derivative_unaries = d_unaries_to_update

        if d_pairwise_to_update is None:
            derivative_pairwise = sharedX(np.zeros((self.batch_size, self.num_indexes, self.num_labels, self.num_indexes, self.num_labels), config.floatX))
        else:
            derivative_pairwise = d_pairwise_to_update

        """
        Fill the pairwise potentials derivatives.
        Does an equivalent of:
        for i in index:
           li = outputs[:, i]
           for v in neighbors(i):
               lv = outputs[:, v]
               derivative[:, i, li, v, lv] += 1
        """
        def fill_pairwise_derivative_for_neighbor(neighbor_index, P_pairwise_d_current, index, label_index, outputs):
            label_neighbor = outputs[:, neighbor_index]
            return T.inc_subtensor(P_pairwise_d_current[:, index, label_index, neighbor_index, label_neighbor], 1)

        def fill_pairwise_derivative(index, neighbors_indexes, neighborhoods_size, P_pairwise_d_current, outputs):
            label_index = outputs[:, index]
            scan_outputs, scan_updates = theano.scan(fn=fill_pairwise_derivative_for_neighbor, sequences=[neighbors_indexes[:neighborhoods_size]], outputs_info=[P_pairwise_d_current], non_sequences=[index, label_index, outputs])
            return scan_outputs[-1], scan_updates

        scan_outputs, scan_updates = theano.scan(fn=fill_pairwise_derivative, sequences=[theano.tensor.arange(self.num_indexes), self.neighbors, self.neighborhoods_sizes], outputs_info=[derivative_pairwise], non_sequences=[outputs], n_steps=self.num_indexes)
        derivative_pairwise = scan_outputs[-1]

        """
        Fill the unary potentials derivatives.
        Does an equivalent of:
        for b in batches:
            derivative[b, :, outputs[b, :]] += 1
        """

        
        scan_outputs, scan_updates2 = theano.scan(fn=lambda batch_index, derivative_unaries_current, outputs: T.inc_subtensor(derivative_unaries_current[batch_index, :, outputs[batch_index, :]], 1),
                                         sequences=[theano.tensor.arange(outputs.shape[0])], outputs_info=derivative_unaries, non_sequences=[outputs])
        derivative_unaries = scan_outputs[-1]
        scan_updates.update(scan_updates2)
        return derivative_unaries, derivative_pairwise, scan_updates

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
        updates : subclass of dictionary specifying the update rules for all shared variables
        """
        """
        Does one iteration of gibbs sampling.
        Does an equivalent of:
        for i in index: #Would be better if index order is random
            # does one update step
            for b in batches:
                sum_P_pairwise_for_i[b] = P_pairwise[b, i, :, list_of_neighbors, outputs[list_of_neighbors]].sum(axis=1)
        """
        def update_case(index, neighbors_indexes, neighborhoods_size, current_output, P_unaries, P_pairwise):
            sum_P_pairwise, update = theano.map(fn=lambda batch_index, index, neighbors_indexes, current_output, P_pairwise: P_pairwise[batch_index, index, :, neighbors_indexes, current_output[batch_index, neighbors_indexes]].sum(axis=1), sequences=[theano.tensor.arange(current_output.shape[0])], non_sequences=[index, neighbors_indexes[:neighborhoods_size], current_output, P_pairwise])
            P_for_labels = T.exp(T.neg(P_unaries[:, index, :] +  sum_P_pairwise))
            probabilities = P_for_labels / T.sum(P_for_labels, axis=1)[:,None] # num_batches x num_labels
            update_case = self.theano_rng.multinomial(pvals=probabilities)
            update_case = T.argmax(update_case, axis=1) # num_batches
            new_output = T.set_subtensor(current_output[:, index], update_case)
            return new_output, update
        scan_outputs, scan_updates = theano.scan(fn=update_case, sequences=[theano.tensor.arange(self.num_indexes), self.neighbors, self.neighborhoods_sizes], outputs_info=[current_output], non_sequences=[P_unaries, P_pairwise], n_steps=self.num_indexes)
        return scan_outputs[-1], scan_updates
