"""
This module contains cost functions to use with a MLP and a CRF in serial
(pylearn2.models.mlpcrf).
"""

__authors__ = [""]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = [""]
__license__ = "3-clause BSD"
__maintainer__ = ""


import numpy as np
import logging
import warnings

from theano.compat.python2x import OrderedDict
from theano.sandbox.neighbours import images2neibs
from theano import config
from theano import tensor as T

import pylearn2
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import (
    FixedVarDescr, DefaultDataSpecsMixin, NullDataSpecsMixin
)
from pylearn2.models import mlpcrf
from pylearn2 import utils
from pylearn2.utils import make_name
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils.rng import make_theano_rng
import theano


logger = logging.getLogger(__name__) 

class PseudoLikelihood(Cost):
    """
    Parameters
    ----------
    theano_rng : MRG_RandomStreams, optional
        If specified, uses this object to generate all random numbers.
        Otherwise, makes its own random number generator.
    """
    def __init__(self, theano_rng=None):
        self.supervised = True
        self.__dict__.update(locals())
        del self.self
        self.theano_rng = make_theano_rng(theano_rng, 2014+8+7,
                which_method="binomial")

    def expr(self, model, data):
        """
        .. todo::

            WRITEME

        The partition function makes this intractable.
        """
        self.get_data_specs(model)[0].validate(data)

        X, Y = data
        assert Y is not None

        Y = Y.reshape((model.batch_size, model.output_shape[0]*model.output_shape[1]))
        Y_u = Y[:, model.indexes_reshaped]
        Y_v = Y[:, model.indexes_neighbors_reshaped]
        Y_edges = model.num_labels*Y_u + Y_v

        P_unaries, P_pairwise = model.get_potentials(X)

        # P_unaries needs to be of size (num_batches, num_indexes, num_labels)
        P_unaries = P_unaries.dimshuffle((3, 1, 2, 0))
        P_unaries = P_unaries.reshape((model.batch_size, model.num_indexes, model.num_labels))
        
        # P_pairwise needs to be of size (num_batches, num_indexes*num_neighbors, num_labels**2)
        P_pairwise = P_pairwise.dimshuffle((4, 2, 3, 0, 1))
        P_pairwise = P_pairwise.reshape((model.batch_size, model.num_indexes*model.num_neighbors, model.num_labels**2))

        E_positve = self.compute_positive_energy(Y, Y_edges, P_unaries, P_pairwise, model.num_labels, model.batch_size)
        E_negative = self.compute_negative_energy(Y_v, P_unaries, P_pairwise, model.num_labels, model.batch_size, model.num_indexes, model.num_neighbors)

        return (E_positve + E_negative)

    def get_monitoring_channels(self, model, data):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()
        rval['CRF_pseudo_likelihood'] = self.expr(model, data)

        return rval

    @wraps(Cost.get_data_specs)
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()

    def one_hot(self, t, r):
        """
        given a tensor t of dimension d with integer values from range(r), return a
        new tensor of dimension d + 1 with values 0/1, where the last dimension
        gives a one-hot representation of the values in t.
        """
        ranges = T.shape_padleft(T.arange(r), t.ndim)
        return T.eq(ranges, T.shape_padright(t, 1))

    def compute_positive_energy(self, output, output_edges, P_unaries, P_pairwise, num_labels, num_batches):
        # compute positive energy unary part
        M_positive_u = self.one_hot(output, num_labels)
        E_positive_u = M_positive_u*P_unaries
        E_positive_u = T.sum(E_positive_u)

        # compute positive energy pairwise part
        M_positive_p = self.one_hot(output_edges, num_labels**2)
        E_positive_p = M_positive_p*P_pairwise
        E_positive_p = T.sum(E_positive_p)

        # compute positive part of the energy
        return (E_positive_u + E_positive_p)/num_batches

    def compute_negative_energy(self, output_v, P_unaries, P_pairwise, num_labels, num_batches, num_indexes, num_neighbors):
        # compute negative energy, unary part
        E_negative_u = P_unaries

        # prepare a martix of 1 to compute the pairwise potential parts
        M_negative_p = self.one_hot(output_v, num_labels)
        M_negative_p= T.tile(M_negative_p, (1, 1, num_labels))

        # compute the pairwise potential according to the state of the neighborhood
        Pp_negative = P_pairwise*M_negative_p

        # compute the pairwise sum over the neighborhood
        Pp_reshape = Pp_negative.reshape((Pp_negative.shape[0], 1, Pp_negative.shape[1], Pp_negative.shape[2]))
        local_values = images2neibs(Pp_reshape, (num_neighbors, num_labels))
        E_negative_p = local_values.sum(axis=1)
        E_negative_p = E_negative_p.reshape((num_batches, num_indexes, num_labels))

        # compute the negative part of the energy
        return (T.log(T.exp(-E_negative_u-E_negative_p).sum(axis=2)).sum(axis=1)).sum(axis=0)/num_batches
