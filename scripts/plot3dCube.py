import numpy as np
import sys

def sigmoid(x, s=0.8, k=0.1):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + np.exp( (-x + s) / k))
    return s

if len(sys.argv) != 2:
    print("Run with arguments <csv_path>")
    exit()

data_path = sys.argv[1]

## Load data using numpy
full = np.genfromtxt(data_path, delimiter=",")

## Get the axis span
z = np.unique(full[:, 0])
y = np.unique(full[:, 1])
x = np.unique(full[:, 2])

print("X:", x)
print("Y:", y)
print("Z:", z)

## Create the data matrix for easier data handling
times = np.zeros((len(x)* len(y)* len(z)))

## Load the actual times
for i, time in enumerate(full[:,-1]):
    times[i] = time

## Transform data from 1D -> 3D
times = times.reshape(len(z),  len(y), len(x))
for i in range(times.shape[0]):
    slice = times[i,:,:].copy()
    slice_normalized = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
    times[i, :,:,] = slice_normalized

## Repeat each data point <rep> times to make each pixel larger
## Trick to improve visualization
rep = 20
times = np.repeat(times, rep, axis=0)
times = np.repeat(times, rep, axis=1)
times = np.repeat(times, rep, axis=2)

## Actual plot

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Extract the color map function
import matplotlib
cmap = matplotlib.cm.get_cmap('viridis')


data = times
## Color array
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
## Alpha
# Option 1: Fixed alpha
#d2[..., 3] = 4

# Option 2: Data dependent
#strength = 10  # <-- Filters out lower values, a larger value is a more agressive filter

# polinomio
#d2[..., 3] = np.power(1 - data/data.max(), strength) * (255.)
# sigmoid
d2[..., 3] = sigmoid(1 - data, s=0.999, k=0.002) * (200.)

## Color values
# Filter helps to distribute the color map values among the values that are shown
# (with significant alpha)

filter = np.percentile(data, 10) # Divide the cmap among the 10th percentile of values 
print(filter)
data[data > filter] = filter # Filter the rest

# Renormalize filtered values to [0, 1] (the 10th percentile)
data = (data- np.min(data)) / (np.max(data) - np.min(data))
# Option 1: Scale
d2[..., 0] = cmap(data)[...,0] * 255#0#255
d2[..., 1] = cmap(data)[...,1] * 255#0#255
d2[..., 2] = cmap(data)[...,2] * 255#0#255

# Option 2: Fixed color
#d2[..., 0] = 0x00
#d2[..., 1] = 0x99
#d2[..., 2] = 0xff



## Uncomment to paint the axes
#d2[:, 0, 0] = [255,0,0,255]
#d2[0, :, 0] = [0,255,0,255]
#d2[0, 0, :] = [0,0,255,255]


"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""
from pyqtgraph.Qt import QtCore, QtGui,QtWidgets

## Background color
pg.setConfigOption('background', (200,200,200,255))


app = QtWidgets.QApplication([])
w = gl.GLViewWidget()
# Switch to 'nearly' orthographic projection.
w.opts['distance'] = 2000
w.opts['fov'] = 1
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')


## Bottom grid
g = gl.GLGridItem(glOptions='opaque', color=(0,0,0,255))
g.translate(50,50,0)
g.scale(10, 10, 1)
g.setDepthValue(2)
w.addItem(g)


## Volume data
v = gl.GLVolumeItem(d2)
v.scale(0.5,0.5,0.5)
v.setDepthValue(2)
w.addItem(v)


## Text
def genText(pos, color, text):
    t = gl.GLTextItem()
    t.setData(pos=pos, color=color, text=text)
    t.scale(10,10,10)
    return t
#                  x, y, z
w.addItem(genText((2, 0, 0), (255, 0, 0, 255), "X"))
w.addItem(genText((0, 2, 0), (0, 255, 0, 255), "Y"))
w.addItem(genText((0, 0, 2), (0, 0, 255, 255), "Z"))


## Axis lines, not very visible
ax = gl.GLAxisItem(glOptions='opaque')
ax.setSize(x=10, y=10,z=10)
ax.scale(10,10,10)
w.addItem(ax)

if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtWidgets.QApplication.instance().exec_()

