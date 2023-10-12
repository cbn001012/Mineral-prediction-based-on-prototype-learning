'''
    This file is used to plot a PA (Prediction Area) image
'''

import numpy as np
import matplotlib.pyplot as plt

def processingLocationPadding1(location:list):
    height = 101
    width = 107
    add = location
    for data in location:
        i,j = data[0],data[1]
        if((i>0 and i<height-1) and (j>0 and j<width-1)):
            add1 = [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
            add = add + add1
        elif i==0 and j!=0 and j!=width-1:
            add2 = [(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
            add = add + add2
        elif i==height-1 and j!=0 and j!=width-1:
            add3 = [(i,j-1),(i,j+1),(i-1,j-1),(i-1,j),(i-1,j+1)]
            add = add + add3
        elif j==0 and i!=0 and i!=height-1:
            add4 = [(i-1,j),(i-1,j+1),(i,j+1),(i+1,j),(i+1,j+1)]
            add = add + add4
        elif j==width-1 and i!=0 and i!=height-1:
            add5 = [(i-1,j),(i-1,j-1),(i,j-1),(i+1,j),(i+1,j-1)]
            add = add + add5
        elif i==0 and j==0:
            add6 = [(i,j+1),(i+1,j),(i+1,j+1)]
            add = add + add6
        elif i==0 and j==width-1:
            add7 = [(i,j-1),(i+1,j-1),(i+1,j)]
            add = add + add7
        elif i==height-1 and j==0:
            add8 = [(i-1,j),(i-1,j+1),(i,j+1)]
            add = add + add8
        else:
            add9 = [(i-1,j-1),(i-1,j),(i,j-1)]
            add = add + add9
    return list(set(add))

# Load the coordinates for four types of minerals
Hydrothermal = [(31,35),(19,50),(20,49),(14,84)]
Porphyry = [(52,11),(25,49),(19,60)]
Skarn = [(61,35),(74,39),(22,71),(20,54),(25,51),(17,54),(25,53),(20,55),(22,49),(21,47),(72,23),(72,24)]
Volcano = [(48,13),(66,15),(52,11),(61,1),(51,10),(85,52),(53,16),(58,5),(56,5),(55,13),(57,18),(70,15),(58,16),
           (54,4),(53,14),(63,14),(56,17),(58,10),(49,16),(54,13),(89,49),(70,55),(87,50),(80,54),(83,50),(60,49),
           (52,12)]

positive_location = Hydrothermal+Porphyry+Skarn+Volcano
print(len(positive_location))
positive_location = processingLocationPadding1(positive_location)
print(len(positive_location))

# 1:Prototypical Network;
# 2:Geo-Meta ;
# 3:Relation Network ;
# 4:Matching Network ;
net_type = 3

if net_type==1:
    file_name = "ProtoNet"
elif net_type==2:
    file_name = "Geo-Meta"
elif net_type==3:
    file_name = "RelationNet"
elif net_type==4:
    file_name = "MatchingNet"

data_path1 = "numpyData/{}_Hydrothermal_img.npy".format(file_name)
data_path2 = "numpyData/{}_Porphyry_img.npy".format(file_name)
data_path3 = "numpyData/{}_Skarn_img.npy".format(file_name)
data_path4 = "numpyData/{}_Volcano_img.npy".format(file_name)

data1 = np.load(data_path1)
data2 = np.load(data_path2)
data3 = np.load(data_path3)
data4 = np.load(data_path4)

data_sum = data1+data2+data3+data4
data_sum = (data_sum-np.min(data_sum))/(np.max(data_sum)-np.min(data_sum))
width = data_sum.shape[0]
high = data_sum.shape[1]

total = []
positive = []

for data_row in data_sum:
    for data in data_row:
        total.append(data)

for (i,j) in positive_location:
    positive.append(data_sum[i][j])

total_area = len(total)
predictionClaCorPer = []
AreaClaCorPer = []

for i in np.linspace(0, 1.0, 101):
    predictionClaCorPer.append(100.0 * len(np.where(positive>i)[0])/len(positive))
    AreaClaCorPer.append((100.0 * len(np.where(total>i)[0])/total_area))

custom_y_left = []
for i in range(11):
    custom_y_left.append(str(i*10)+'%')
custom_y_right = custom_y_left[::-1]

print(len(predictionClaCorPer))
print(len(AreaClaCorPer))

fig, ax1 = plt.subplots(figsize=(8, 8), dpi=100)

ax1.grid(axis='both', linestyle='-.')


ax1.plot(np.linspace(0, 1.0, 101), predictionClaCorPer, color="red", alpha=0.5, label="Prediction rate",linewidth= 2)
ax1.set_yticks(np.linspace(0, 100, 11), custom_y_left)    
plt.ylim((0-0.2,100+0.2))
ax1.set_xticks(np.linspace(0, 1.0, 11))   
ax1.set_xlabel('Prospectivity score', fontdict={'size': 16})
ax1.set_ylabel('Percentage of known mine occurrences', fontdict={'size': 16})
ax2 = ax1.twinx()
ax2.set_yticks(np.linspace(100, 0, 11),custom_y_right)   
plt.ylim((0-0.2,100+0.2))
ax2.invert_yaxis()                          
ax2.set_ylabel('Percentage of study area', fontdict={'size': 16})
ax2.plot(np.linspace(0, 1.0, 101), AreaClaCorPer, color="green", alpha=0.5, label="Area",linewidth=2)

fig.legend(loc=4, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
plt.title('{}'.format(file_name))
plt.show(block=True)


