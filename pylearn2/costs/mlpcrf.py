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
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
RandomStreams = MRG_RandomStreams
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
from pylearn2.utils.rng import make_theano_rng


logger = logging.getLogger(__name__) 

def ConstrastiveDivergence(Cost):
    """
    Parameters
    ----------
    num_gibbs_steps : int
        The number of Gibbs steps to use in the negative phase. (i.e., if
        you want to use CD-k or PCD-k, this is "k").
    theano_rng : MRG_RandomStreams, optional
        If specified, uses this object to generate all random numbers.
        Otherwise, makes its own random number generator.
    """
    def __init__(self, num_gibbs_steps, theano_rng=None):
        self.supervised = True #TODO: check it is needed to set it
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

        return None

    def get_monitoring_channels(self, model, data):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        rval = OrderedDict()

        X, Y = data
        #TODO ?

        return rval

    def get_gradients(self, model, data):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        X, Y = data
        assert Y is not None

        P_unaries, P_pairwise = model.get_potentials(X)

        d_unaries_pos, d_pairwise_pos = self._get_positive_phase(model, Y)

        d_unaries_neg, d_pairwise_neg = self._get_negative_phase(model, P_unaries, P_pairwise, Y)

        d_unaries_estimate = d_unaries_pos + d_unaries_neg
        d_pairwise_estimate = d_pairwise_pos + d_pairwise_neg

        updates = OrderedDict()
        updates[d_unaries] = d_unaries_estimate
        updates[d_pairwise] = d_pairwise_estimate

        gradients = OrderedDict()
        gradients = model.propagate_gradient(gradients, d_unaries, d_pairwise)

        return gradients, updates

    def _get_positive_phase(model, Y):
        """
        .. todo::

            WRITEME
        """
        return self.model.calculate_derivates_energy(Y).mean(axis=?)

    def _get_negative_phase(model, P_unaries, P_pairwise, Y):
        """
        .. todo::

            WRITEME
        """
        #TODO
