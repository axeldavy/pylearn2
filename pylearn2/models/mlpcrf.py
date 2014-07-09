"""
This class implements an MLP followed by a CRF.
"""

import functools
import logging
import numpy as np
import warnings

from theano.compat.python2x import OrderedDict
from theano.sandbox import cuda
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
       
    def calculate_derivates_energy(self, outputs):
        #TODO
        
    def gibbs_sample_step(self, P_unaries, P_pairwise, current_output):
        #TODO