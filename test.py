import random
from array import array
output_file = open('heightmap.dhm', 'wb')
int_array = array('i', [512, 512])
int_array.tofile(output_file)

floats = []
for i in range(512*512):
    floats.append(random.random())

float_array = array('f', floats)
float_array.tofile(output_file)
output_file.close()
