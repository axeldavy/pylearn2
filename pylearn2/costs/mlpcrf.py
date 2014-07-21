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

class ConstrastiveDivergence(Cost):
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

    @wraps(Cost.get_data_specs)
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()

    def get_gradients(self, model, data):
        """
        .. todo::

            WRITEME
        """
        self.gibbs_var = theano.shared(np.zeros((self.num_gibbs_steps, model.batch_size, model.num_indexes), dtype = np.int)) 
        self.get_data_specs(model)[0].validate(data)
        X, Y = data
        assert Y is not None

        P_unaries, P_pairwise, get_potentials_updates = model.get_potentials(X)

        pos_phase_energy, pos_updates = self._get_positive_phase(model, P_unaries, P_pairwise, Y)

        neg_phase_energy, neg_updates = self._get_negative_phase(model, P_unaries, P_pairwise, Y)

        params = list(model.get_params())

        gradients = OrderedDict(
            safe_zip(params, T.grad(pos_phase_energy + neg_phase_energy,
                                    params, consider_constant=[self.gibbs_var],
                                    disconnected_inputs='ignore'))
            )

        updates = OrderedDict()
        for key, val in get_potentials_updates.items():
            updates[key] = val
        for key, val in pos_updates.items():
            updates[key] = val
        for key, val in neg_updates.items():
            updates[key] = val

        return gradients, updates

    def _get_positive_phase(self, model, P_unaries, P_pairwise, Y):
        """
        .. todo::

            WRITEME
        """
        positive_energy, positive_updates = model.calculate_energy(P_unaries, P_pairwise, Y)

        return positive_energy, positive_updates

    def _get_negative_phase(self, model, P_unaries, P_pairwise, Y):
        """
        .. todo::

            WRITEME
        """
        def call_gibbs_sampling_step(Y, P_unaries, P_pairwise):
            next_Y, next_Y_updates = model.gibbs_sample_step(P_unaries, P_pairwise, Y)
            return next_Y, next_Y_updates

        def compute_energy_for_samples(Y, P_unaries, P_pairwise):
            return model.calculate_energy(P_unaries, P_pairwise, Y)

        samples_Y_outputs, samples_Y_updates = theano.scan(fn=call_gibbs_sampling_step, outputs_info=[Y], non_sequences=[P_unaries, P_pairwise], n_steps=self.num_gibbs_steps)
        samples_energies_outputs, samples_energies_updates = theano.map(fn=compute_energy_for_samples, sequences=[self.gibbs_var], non_sequences=[P_unaries, P_pairwise])
        
        updates = OrderedDict()
        updates[self.gibbs_var] = samples_Y_outputs
        for key, val in samples_Y_updates.items():
            updates[key] = val
        for key, val in samples_energies_updates.items():
            updates[key] = val

        return -samples_energies_outputs.mean(), updates