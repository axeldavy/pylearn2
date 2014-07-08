import os
#import ipdb
import logging
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets.dataset import Dataset
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.iteration import SequentialSubsetIterator, resolve_iterator_class
from itertools import izip
from lisa_brats.brains import BrainSet


class BRATS(Dataset):
    """
    TODO
    """
    def __init__(self, path_brains, path_analysis, num_minibatches_train=None, num_minibatches_test=None, axes = ('b', 0, 1, 'c'), max_labels=None, distribution=None):

        self.__dict__.update(locals())
        del self.self

        # load data
        path_brains = preprocess(path_brains)
        path_analysis = preprocess(path_analysis)

        self.brain_set = BrainSet.from_path(path_brains)
        self.h5file = tables.openFile(path_analysis, mode = 'r')
        if distribution is not None:
            assert len(distribution) == 5
            distribution = numpy.asarray(distribution, dtype=numpy.int)
            # FIXME for now don't use floats.
            assert distribution.sum() == 128
        self.distribution = distribution
        self.labels_tables = []
        self.num_minibatches_train = num_minibatches_train
        self.num_minibatches_test = num_minibatches_test
        self.lengths = numpy.zeros((5,3), dtype=numpy.int)
        self.current_index_training = numpy.zeros((5,3), dtype=numpy.int)
        self.current_index_testing = numpy.zeros((5,3), dtype=numpy.int)
        for i in range(5):
            self.labels_tables.append([])
            for j, name in izip(xrange(3), ['all', 'nonnullentropy32', 'nonnullentropy8']):
                table = self.h5file.getNode('/' + str(i), name=name)
                self.labels_tables[i].append(table)
                self.lengths[i,j] = table.shape[0]

        super(BRATS, self).__init__()

        self.h5file.flush()

    def get_patch(self, brain_name, z, x, y):
        brain = self.brain_set.get_brain_by_name(brain_name)
        images = brain.images
        labels = brain.labels
        patch_shape = (32, 32)
        label_onehot_patch = numpy.zeros((32*32,5))
        x_left = x - patch_shape[0]/2
        x_right = x_left + patch_shape[0]
        y_left = y - patch_shape[1]/2
        y_right = y_left + patch_shape[1]
        #z_left = z - patch_shape[0]/2
        #z_right = z_left + patch_shape[0]
        slice_x = slice(x_left, x_right)
        slice_y = slice(y_left, y_right)
        #slice_z = slice(z_left, z_right)
        im_patch = images[z, slice_x, slice_y, ...]
        label_patch = labels[z, slice_x, slice_y, ...]
        for i,l in enumerate(label_patch.flatten()):
            label_onehot_patch[i,l] = 1
            
        # image3orth: we store the orthogonal patches in additional channels
        #orth1 = images[slice_z, x, slice_y, ...]
        #orth2 = images[slice_z, slice_x, y, ...]
        #print im_patch.shape,orth1.shape,orth2.shape, images.shape, x_left, x_right, y_left, y_right, z_left, z_right
        return (im_patch.swapaxes(0,2), label_onehot_patch)

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                topo=None, targets=None, rng=None, return_tuple=True,
                data_specs=None):
        """
        method inherited from Dataset
        """
        #if hasattr(self, 'subset_iterator'):
        #    return self
        #self.mode = mode
        #self.batch_size = batch_size
        #self._targets = targets
        
        
        """if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        if rng is not None:
            mode = 'random_slice'"""
        
        if batch_size <> 128:
            raise ValueError("only a batch_size of 128 is supported")
        self.count = 0
        if self.distribution is None:
            self.num_examples_labels = numpy.ones((5,3), dtype=numpy.int) * 8
            self.num_examples_labels[0,0] = 16
        else:
            #TODO use real distribution:
            self.num_examples_labels = numpy.zeros((5,3), dtype=numpy.int)
            for i in xrange(5):
                num_for_label = self.distribution[i]
                dec = num_for_label//3
                self.num_examples_labels[i,0] = dec
                self.num_examples_labels[i,1] = dec
                self.num_examples_labels[i,2] = num_for_label - 2 * dec
            assert self.num_examples_labels.sum() == 128
        if rng is None:
            self.current_index_testing = numpy.zeros((5,3), dtype=numpy.int)
            self.stochastic = False
            if self.num_minibatches_test is None:
                raise ValueError("not able to be used for testing: num_minibatches_test not set")
        else:
            self.stochastic = True
            if self.num_minibatches_train is None:
                raise ValueError("not able to be used for training: num_minibatches_train not set")
        return self


    def __iter__(self):
        return self

    def next(self):
        if (self.stochastic and self.count >= self.num_minibatches_train) or ((not (self.stochastic)) and self.count >= self.num_minibatches_test):
            raise StopIteration()
        X0 = numpy.zeros((4, 32, 32, 128), dtype=numpy.float32)
        #X1 = numpy.zeros((4, 32, 32, 128), dtype=numpy.float32)
        #X2 = numpy.zeros((4, 32, 32, 128), dtype=numpy.float32)
        y = numpy.zeros((32*32,5,128), dtype=numpy.float32)
        index = 0
        if self.stochastic:
            current_index_array = self.current_index_training.copy()
        else:
            current_index_array = self.current_index_testing.copy()
        for i in xrange(5):
            for j in xrange(3):
                num_examples = self.num_examples_labels[i,j]
                current_index = current_index_array[i,j]
                data = self.labels_tables[i][j].read(start=current_index, stop=current_index+num_examples)
                current_index_array[i,j] += num_examples
                assert len(data) <= num_examples
                if len(data) < num_examples:
                    current_index_array[i,j] = num_examples-len(data)
                    data = numpy.concatenate((data, self.labels_tables[i][j].read(start=0, stop=num_examples-len(data))), axis=0)
                assert len(data) == num_examples
                for d in data:
                    #y[index, i] = 1.
                    i0 = self.get_patch(d[0], d[1], d[2], d[3])
                    X0[..., index] = i0[0]
                    y[...,index] = i0[1]
                    #X1[..., index] = i1
                    #X2[..., index] = i2
                    index += 1
        
        y = y.reshape(32*32*5,128)
        y = y.T
        #y = y.reshape(128*32*32,5)
        
        mini_batch = (X0, y)
        assert index == 128
        self.count += 1
        if self.stochastic:
            self.current_index_training = current_index_array
        else:
            self.current_index_testing = current_index_array
        return mini_batch
        
    @property
    def batch_size(self):
        """
        .. todo::

            WRITEME
        """
        return 128

    @property
    def num_batches(self):
        """
        .. todo::

            WRITEME
        """
        if self.stochastic:
            return self.num_minibatches_train
        else:
            return self.num_minibatches_test

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        if self.stochastic:
            return self.num_minibatches_train * 128
        else:
            return self.num_minibatches_test * 128

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return False
