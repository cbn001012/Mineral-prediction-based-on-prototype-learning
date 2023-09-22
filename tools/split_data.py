from skimage import transform,io

pixSize = 10    # 切割图像的大小(长宽),切割图像是正边形
filePath = '/mnt/3.6T-DATA/CBN/DATA/dim10_chang.tif'
trainHydrothermalDataSavePath = './dim/image/hydrothermal/'
trainNegativeDataSavePath = './dim/image/negative/'
trainPorphyryDataSavePath = './dim/image/porphyry/'
trainSkarnDataSavePath = './dim/image/skarn/'
trainVolcanoDataSavePath = './dim/image/volcano/'
# trainPositivePath = '/DATA/dim10_chang_0107/positive/'
# trainNegativePath = '/DATA/dim10_chang_0107/negative/'


#
# valHydrothermalDataSavePath = './validation/hydrothermal/'
# valNegativeDataSavePath = './validation/negative/'
# valPorphyryDataSavePath = './validation/porphyry/'
# valSkarnDataSavePath = './validation/skarn/'
# valVolcanoDataSavePath = './validation/volcano/'

# 图片加载
im = io.imread(filePath)

sp = im.shape
height = int(sp[0]/pixSize)
width = int(sp[1]/pixSize)

# (行，列) 从0开始
positive = [(3,72),(7,22),(14,10),(11,13),(15,13),(15,14),(21,14),(10,17),(21,17),(12,22),(20,22),(12,24),(11,31),(16,50)] #长方形坐标
# positive = [(31,35),(19,50),(20,49),(14,84),(52,11),(25,49),(19,60),(61,35),(74,39),
#             (22,71),(20,54),(25,51),(17,54),(25,53),(20,55),(22,49),(21,47),(13,23),
#             (12,24),(48,13),(66,15),(52,11),(61,1),(51,10),(85,52),(53,16),(58,5),
#             (56,5),(55,13),(57,18),(70,15),(58,16),(54,4),(53,14),(63,14),(56,17),
#             (58,10),(49,16),(54,13),(89,49),(70,55),(87,50),(80,54),(83,50),(60,49),
#             (52,12)]
# 46个正样本
# 找出周围图片的坐标
def processingLocationPadding1(location:list):
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
    return list(set(add))   # 列表去重

def processingLocationPadding2(location:list):
    # add存放所有拓展，min存放不合规的拓展
    add = location
    for data in location:
        i,j = data[0],data[1]
        add2 = [(i-2,j-2),(i-2,j-1),(i-2,j),(i-2,j+1),(i-2,j+2),
                (i-1,j-2),(i-1,j-1),(i-1,j),(i-1,j+1),(i-1,j+2),
                (i,j-2),(i,j-1),(i,j+1),(i,j+2),
                (i+1,j-2),(i+1,j-1),(i+1,j),(i+1,j+1),(i+1,j+2),
                (i+2,j-2),(i+2,j-1),(i+2,j),(i+2,j+1),(i+2,j+2)]
        add = add + add2

    add = list(set(add))    # 列表去重

    min = []
    for k in range(len(add)):
        i,j = add[k]
        if (i<0 or i>height-1 or j<0 or j>width-1):
            min.append((i,j))

    L = [x for x in add if x not in min]
    L = list(set(L))
    return L


# aft_processingPositive = processingLocationPadding1(positive)
aft_processingPositive = positive
aft_negative = aft_processingPositive
aft_negative = list(set(aft_negative))

iter_num = 0
positiveNum = 0
negativeNum = 0

for i in range(height):            # 高，纵向
    # if i<= int(height/2):       # 处理训练集，上半部分图片
    #     pass
    for j in range(width):         # 宽，横向
        print("processing Train img No "+str(iter_num)+" image")

        ig = im[i*pixSize:(i+1)*pixSize, j*pixSize:(j+1)*pixSize]    # 滑动窗口

        if (i,j) in aft_processingPositive: # 正样本
            io.imsave(trainPositivePath + str(iter_num) + '.tif',ig)
            positiveNum = positiveNum + 1

        if (i,j) not in aft_negative :  # 负样本
            io.imsave(trainNegativePath + str(iter_num) + '.tif',ig)
            negativeNum += 1
        iter_num += 1

print("正样本共：%d 个"%(positiveNum))
print("负样本共：%d 个"%(negativeNum))