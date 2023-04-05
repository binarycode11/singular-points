import numpy
import numpy as np
a = numpy.arange(10)
include_index = np.arange(4)
mask = np.ones(len(a), dtype=bool) # all elements included/True.
mask[include_index] = False              # Set unwanted elements to False

print(a[mask],a[~mask])
