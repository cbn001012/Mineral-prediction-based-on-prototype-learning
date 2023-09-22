import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import matplotlib.cm as cm


# 1:protoNet;
# 2:ss-protoNet ;
# 3:relationNet ;
# 4:marchNet ;
net_type = 3

if net_type==1:
    file_name = "protoNet"
elif net_type==2:
    file_name = "ss-protoNet"
elif net_type==3:
    file_name = "ProtoGCN"
elif net_type==4:
    file_name = "marchNet"

# 加载热力图
Hydrothermal_img = np.load("numpyData/dim39/{}_Hydrothermal_img.npy".format(file_name))
Porphyry_img = np.load("numpyData/dim39/{}_Porphyry_img.npy".format(file_name))
Skarn_img = np.load("numpyData/dim39/{}_Skarn_img.npy".format(file_name))
Volcano_img = np.load("numpyData/dim39/{}_Volcano_img.npy".format(file_name))

all_img = Hydrothermal_img+Skarn_img+Porphyry_img+Volcano_img
all_img = (all_img-all_img.min())/(all_img.max()-all_img.min())
# all_img = np.array(all_img,dtype=np.float32)

# 放大N倍
def fun(img, N=10):
    h,w = img.shape
    new_img = np.zeros((h*N, w*N), dtype=np.float32)
    print(new_img.shape)
    for i in range(h):
        for j in range(w):
            for x in np.linspace(i*N,i*N+N-1,N):
                for y in np.linspace(j*N,j*N+N-1,N):
                    new_img[int(x)][int(y)] = img[i][j]
    return new_img


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
# plt.subplots(figsize=(8, 8), dpi=100)
plt.imshow(all_img, interpolation='None', vmin=0, vmax=1, cmap=cm.get_cmap('jet'))
plt.colorbar()
plt.title('{}_All'.format(file_name))
plt.axis('off')
plt.show(block=True)

