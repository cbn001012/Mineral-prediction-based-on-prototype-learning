import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.cm as cm


# 1:Prototypical Network;
# 2:Geo-Meta ;
# 3:Relation Network ;
# 4:Matching Network ;
net_type = 2

if net_type==1:
    file_name = "ProtoNet"
elif net_type==2:
    file_name = "Geo-Meta"
elif net_type==3:
    file_name = "RelationNet"
elif net_type==4:
    file_name = "MatchingNet"

# Load the numpy data of heatmaps
Hydrothermal_img = np.load("../numpyData/{}_Hydrothermal_img.npy".format(file_name))
Porphyry_img = np.load("../numpyData/{}_Porphyry_img.npy".format(file_name))
Skarn_img = np.load("../numpyData/{}_Skarn_img.npy".format(file_name))
Volcano_img = np.load("../numpyData/{}_Volcano_img.npy".format(file_name))

all_img = Hydrothermal_img+Skarn_img+Porphyry_img+Volcano_img
all_img = (all_img-all_img.min())/(all_img.max()-all_img.min())

# creates four Axes
fig, axes = plt.subplots(figsize=(10, 8), dpi=100, nrows=2, ncols=2)

norm = colors.Normalize(vmin=0, vmax=1)
a = axes[0][0].imshow(Hydrothermal_img, vmin=0, vmax=1, cmap=cm.get_cmap('jet'))
axes[0][0].axis('off')
b = axes[0][1].imshow(Skarn_img, vmin=0, vmax=1, cmap=cm.get_cmap('jet'))
axes[0][1].axis('off')
c = axes[1][0].imshow(Porphyry_img, vmin=0, vmax=1, cmap=cm.get_cmap('jet'))
axes[1][0].axis('off')
d = axes[1][1].imshow(Volcano_img, vmin=0, vmax=1, cmap=cm.get_cmap('jet'))
plt.colorbar(a, ax=axes.ravel().tolist())
axes[0][0].set_title('{}_Hydrothermal'.format(file_name))
axes[0][1].set_title('{}_Skarn'.format(file_name))
axes[1][0].set_title('{}_Porphyry'.format(file_name))
axes[1][1].set_title('{}_Volcano'.format(file_name))
axes[1][1].axis('off')
plt.show(block=True)

interpolation='spline16'
plt.imshow(all_img, interpolation='None', vmin=0, vmax=1, cmap=cm.get_cmap('jet'))
plt.colorbar()
plt.title('{}_All'.format(file_name))
plt.axis('off')
plt.show(block=True)

