import numpy as np
import sys

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


## Extracts a color map function using the specified list of colors.
def extract_color_map(colors, cmap_name='custom_cmap'):
    cmap = mcolors.ListedColormap(colors, name=cmap_name)
    return cmap

## Define a list of RGB tuples representing the color list
## a value will fall into, ordered linearly
colors = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]

# Extract the color map function
cmap = extract_color_map(colors)


data = times

## Color
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
# Option 1: Scale
#d2[..., 0] = cmap(data)[...,0] * 255#0#255
#d2[..., 1] = cmap(data)[...,1] * 255#0#255
#d2[..., 2] = cmap(data)[...,2] * 255#0#255

# Option 2: Fixed color
d2[..., 0] = 0x00
d2[..., 1] = 0x99
d2[..., 2] = 0xff


## Alpha
# Option 1: Fixed alpha
#d2[..., 3] = 4

# Option 2: Data dependent
strength = 1.75  # <-- Filters out lower values, a larger value is a more agressive filter
d2[..., 3] = np.power(data/data.max(), strength) * (255.)

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

