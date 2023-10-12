'''
The code essentially crops smaller images from a larger input image based on the specified pixSize
and saves them separately as positive and negative samples, depending on their coordinates.
'''

from skimage import transform,io

# Size of the cropped image
pixSize = 10

# Define the path of the large image to be cropped
filePath = './DATA/dim10_chang.tif'

# Paths where different categories are saved
trainHydrothermalDataSavePath = './dim/image/hydrothermal/'
trainNegativeDataSavePath = './dim/image/negative/'
trainPorphyryDataSavePath = './dim/image/porphyry/'
trainSkarnDataSavePath = './dim/image/skarn/'
trainVolcanoDataSavePath = './dim/image/volcano/'


im = io.imread(filePath)
sp = im.shape
height = int(sp[0]/pixSize)
width = int(sp[1]/pixSize)

#  (Row, Column) Coordinates starting from 0 that store the positions of positive samples
#  Modify according to the coordinates of your positive samples
positive = [(31,35),(19,50),(20,49),(14,84),(52,11),(25,49),(19,60),(61,35),(74,39),
            (22,71),(20,54),(25,51),(17,54),(25,53),(20,55),(22,49),(21,47),(13,23),
            (12,24),(48,13),(66,15),(52,11),(61,1),(51,10),(85,52),(53,16),(58,5),
            (56,5),(55,13),(57,18),(70,15),(58,16),(54,4),(53,14),(63,14),(56,17),
            (58,10),(49,16),(54,13),(89,49),(70,55),(87,50),(80,54),(83,50),(60,49),
            (52,12)]



def processingLocationPadding1(location:list):
    '''
    Find the coordinates of the eight points surrounding a mineral deposit, mark them as positive samples, and establish a mineral buffer zone
    '''
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

aft_processingPositive = processingLocationPadding1(positive)
aft_negative = aft_processingPositive
aft_negative = list(set(aft_negative))

iter_num = 0
positiveNum = 0
negativeNum = 0

for i in range(height):
    for j in range(width):
        print("processing Train img No "+str(iter_num)+" image")

        ig = im[i*pixSize:(i+1)*pixSize, j*pixSize:(j+1)*pixSize]

        if (i,j) in aft_processingPositive:
            io.imsave(trainPositivePath + str(iter_num) + '.tif',ig)
            positiveNum = positiveNum + 1

        if (i,j) not in aft_negative :
            io.imsave(trainNegativePath + str(iter_num) + '.tif',ig)
            negativeNum += 1
        iter_num += 1

print("total number of positive samples ：%d "%(positiveNum))
print("total number of negative samples ：%d "%(negativeNum))