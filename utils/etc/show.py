import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", dest="inputFile", action="store")
args = parser.parse_args()


# Reading detector configuration
df_geo = pd.read_csv('detector_geometry.csv')

x = np.asarray(df_geo['x'])
y = np.asarray(df_geo['y'])
z = np.asarray(df_geo['z'])


# Reading input data from .npz file
data = np.load(args.inputFile)
# fdata = data[0] # firstTime
# ndata = data[1] # nPE
# print(data['arr_0'])
data = data['arr_0']
fdata = data[:, 0] # firstTime
ndata = data[:, 1] # nPE


# Plot
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# ## Arrow
# idx = np.argmin(fdata)
# X = x[idx]
# Y = y[idx]
# Z = z[idx]
# Zenith = np.pi - data[evt_number][4]
# Azimuth = data[evt_number][5] - np.pi
# dx = np.sin(Zenith) * np.cos(Azimuth)
# dy = np.sin(Zenith) * np.sin(Azimuth)
# dz = np.cos(Zenith)

# ax.quiver(X - 500*dx, Y - 500*dy, Z - 500*dz, dx, dy, dz, length=2000, color='r', linewidth=2, arrow_length_ratio=0.1, normalize=True)


## Detector Shape Convex Hull
edge_string_idx = [1, 6, 50, 74, 73, 78, 75, 31]

top_xy, bottom_xy = [], []
for i in edge_string_idx:
    top_xy.append([x[(i-1)*60], y[(i-1)*60]])
    bottom_xy.append([x[(i-1)*60+59], y[(i-1)*60+59]])

top_xy.append(top_xy[0])
bottom_xy.append(bottom_xy[0])

n = len(top_xy)
z_bottom = -500
z_top = 500
for i in range(n-1):
    x0, y0 = top_xy[i]
    x1, y1 = top_xy[(i + 1)]
    ax.plot([x0, x1], [y0, y1], [z_top, z_top], color='gray')

for i in range(n-1):
    x0, y0 = top_xy[i]
    x1, y1 = top_xy[(i + 1)]
    ax.plot([x0, x1], [y0, y1], [z_bottom, z_bottom], color='gray')

for i in range(n-1):
    _x, _y = top_xy[i]
    ax.plot([_x, _x], [_y, _y], [z_bottom, z_top], color='gray')


## PMT Dots
ax.scatter(x, y, z, s=1, c='gray', alpha=0.5)


## Signals
sc = ax.scatter(x, y, z, s=10*ndata, c=fdata, cmap='viridis', alpha=0.9, depthshade=True)
cbar = fig.colorbar(sc)
cbar.set_label('firstTime (ns)')


## Figure Settings
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')

# Energy = round(data[evt_number, 0], 3)
# X = round(data[evt_number, 1], 3)
# Y = round(data[evt_number, 2], 3)
# Z = round(data[evt_number, 3], 3)
# Zenith = round(np.rad2deg(data[evt_number, 4]), 3)
# Azimuth = round(np.rad2deg(data[evt_number, 5]), 3)

ax.set_title(f"firstTime Response (5160 DOMs)\nEnergy: {Energy} GeV ({Energy/1e6:.3f} PeV)\nZenith: {Zenith:.3f} (deg), Azimuth: {Azimuth:.3f} (deg)\nX: {X} m, Y: {Y} m, Z: {Z} m")

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

ax.dist = 5

plt.tight_layout()
plt.savefig("firstTime_response_3D.png", transparent=True, bbox_inches='tight')
plt.show()