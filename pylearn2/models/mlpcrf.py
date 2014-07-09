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
    def __init__(self, mlp, crf_size, connections):
        #TODO
        
    @wraps(Model.set_input_space)
    def set_input_space(self, input_space):
        #TODO
        
    @wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        #TODO
        
    @wraps(get_monitoring_data_specs)
    def get_monitoring_data_specs(self):
        #TODO
        
    @wraps(set_batch_size)
    def set_batch_size(self):
        #TODO
    
    @wraps(get_lr_scalers)
    def get_lr_scalers(self):
        #TODO
        
    @wraps(redo_theano)
    def redo_theano(self):
        #TODO
       
    def get_potentials(self, inputs):
        #TODO
       
    def calculate_derivates_energy(self, outputs):
        #TODO
        
    def gibbs_sample_step(self, P_unaries, P_pairwise, current_output):
        #TODO