# -*- coding: utf-8 -*-
import gzip
# from .data_utils import get_file
from six.moves import cPickle
import sys

def load_data(path="mnist.pkl.gz"):
  # path = get_file(path, origin="https://s3.amazonaws.com/img-datasets/mnist.pkl.gz")
  if path.endswith(".gz"):
      f = gzip.open(path, 'rb')
  else:
      f = open(path, 'rb')

  if sys.version_info < (3,):
      data = cPickle.load(f)
  else:
      data = cPickle.load(f, encoding="bytes")

  f.close()
  return data  # (X_train, y_train), (X_test, y_test)

'''
checkpoint = tm.time()
(X_train, y_train), (X_test, y_test) = load_data("mnist.pkl")
print "Loaded in %0.2f seconds" % (tm.time() - checkpoint)
print "Number of examples for training: %d rows" % len(X_train)
print "Number of examples for testing: %d rows" % len(X_test)
'''