from __future__ import division, print_function, absolute_import

import numpy as np
from os import listdir
from os.path import isfile, join
import gdal
import gdalconst
from skimage.measure import block_reduce

print("Loading mars data")
hirise_path = "HIRISE"
mars_files = [f for f in listdir(hirise_path) if isfile(join(hirise_path, f))]

mars_data = []
for f in mars_files:
    filename = hirise_path + "/" + f
    print("Loading", filename)
    v = gdal.Open(filename, gdalconst.GA_ReadOnly)
    data = v.ReadAsArray()
    data50 = block_reduce(data, block_size=(20, 20), func=np.mean)
    mars_data.append(data50)
print("Loading mars data done.")

print(mars_data)

np.save("hirise_20", mars_data)
