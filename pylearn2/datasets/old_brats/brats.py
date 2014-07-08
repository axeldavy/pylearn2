import os
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


class BRATS(Dataset):
    """
    TODO
    """
    def __init__(self, path, axes = ('b', 0, 1, 'c'), max_labels=None):

        self.__dict__.update(locals())
        del self.self

        # load data
        path = preprocess(path)

        self.h5file = tables.openFile(path, mode = 'r')
        data = self.h5file.getNode('/', "Data")
        self.X = data.X
        self.y = data.y

        super(BRATS, self).__init__()

        self.h5file.flush()
   
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                topo=None, targets=None, rng=None, return_tuple=True,
                data_specs=None):
        """
        method inherited from Dataset
        """
        #if hasattr(self, 'subset_iterator'):
        #    return self
        self.mode = mode
        #self.batch_size = batch_size
        self._targets = targets
        
        
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        if rng is not None:
            mode = 'random_slice'
        
        mode = resolve_iterator_class(mode)
        
        self.subset_iterator = mode(self.X.shape[0],
                                          batch_size,
                                          num_batches,
                                          rng)
        self.stochastic = self.subset_iterator.stochastic
        return self


    def __iter__(self):
        return self

    def next(self):
        indx = self.subset_iterator.next()
        #print indx
        try:
            mini_batch = (self.X[indx,...].swapaxes(0,3).copy(), self.y[indx,...])
        except IndexError:
            # the ind of minibatch goes beyond the boundary
            import ipdb; ipdb.set_trace()
        return mini_batch
        
    @property
    def batch_size(self):
        """
        .. todo::

            WRITEME
        """
        return self.subset_iterator.batch_size

    @property
    def num_batches(self):
        """
        .. todo::

            WRITEME
        """
        return self.subset_iterator.num_batches

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        return self.subset_iterator.num_examples

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return subset_iterator.uneven