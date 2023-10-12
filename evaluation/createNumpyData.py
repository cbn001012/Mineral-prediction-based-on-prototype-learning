'''
The first step in creating a mineral prediction map is to run the following file.
Input each sample into the trained model, and the model will output the probability of each sample containing minerals. Save the results as an npy file.
'''
from skimage import transform,io
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
import matplotlib.pyplot as plt
from matplotlib import colors
import torchvision.transforms as transforms
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import os
from lib import loadTifImage
from torch_geometric.loader import DataLoader as Graph_Dataloader

class Convnet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.encoder = l2l.vision.models.CNN4Backbone(
            hidden_size=128,
            channels=39,
            max_pool=True,
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Gx(nn.Module):
    def __init__(self, ways):
        super().__init__()
        self.linear1 = nn.Linear(192*2, 512)
        self.linear2 = nn.Linear(512, 1)
        self.ways = ways
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x.reshape(-1, self.ways)

class relationNet(nn.Module):
    def __init__(self, ways):
        super().__init__()
        self.embeddingExtract = Convnet()
        self.Gx = Gx(ways)

    def forward(self,):
        pass

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, data.batch)
        return x

class ProtoGCN(nn.Module):
    def __init__(self, ways):
        super().__init__()
        self.embeddingExtract = Convnet()
        self.Gx = Gx(ways)
        self.GCNembedding = GCN()

    def forward(self,):
        pass

def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)

    return logits

def catEmbeding(a, b):
    n = a.shape[0]
    m = b.shape[0]
    emb = torch.cat([a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)], dim=2)
    return emb

def cosine_score(a, b):
    n = a.shape[0]
    m = b.shape[0]
    cos_score = torch.cosine_similarity(a.unsqueeze(1).expand(n, m, -1),
                                        b.unsqueeze(0).expand(n, m, -1), dim=2)
    return cos_score



device = torch.device("cpu")
print("using {} device.".format(device))

'''
Define the type of network you want to test
net_type:
1: Prototypical Network
2: Geo-Meta
3: Relation Network
4: Matching Network
'''

net_type = 2

model_weight_path = "../saved_model/geometa-ACC_byDL_0.9657.pth"
support = torch.load('../saved_model/geometa-support_ACC_byDL_0.9657.pt')
support = support.to(device)

if net_type == 2:
    net = ProtoGCN(ways=5)
elif net_type == 3:
    net = relationNet(ways=5)
else:
    net = Convnet()

assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path))

net.to(device)
net.eval()

pixSize = 10
filePath = './all_data.tif' # Load all data (uncropped tif format data)

im = io.imread(filePath)
sp = im.shape
height = int(sp[0]/pixSize)
width = int(sp[1]/pixSize)

# numpy-data
Hydrothermal_img = np.zeros((height, width), dtype='float32')
Porphyry_img = np.zeros((height, width), dtype='float32')
Skarn_img = np.zeros((height, width), dtype='float32')
Volcano_img = np.zeros((height, width), dtype='float32')


graph_dataset = torch.load("./test/processed/data.pt")

def get_graph_data(graph_dataset, index):
    data = []
    for graph_data in graph_dataset:
        if graph_data.filename == (str(index)+".tif"):
            data.append(graph_data)
            break
    return data

# Sliding window to generate numpy data
def create_numpy(ig,h,w,graph_data):

    igg = transform.resize(ig, (20, 20))
    igg = igg/255.0
    igg = np.array(igg, dtype=np.float32)
    igg = igg.transpose(2,0,1)
    igg = torch.tensor(igg)
    igg = igg.unsqueeze(0)

    with torch.no_grad():

        net.eval()
        if net_type==1:
            # Prototypical Network
            query = net(igg.to(device))
            outputs = pairwise_distances_logits(query, support)

        elif net_type==2:
            # Geo-Meta
            emb1 = net.embeddingExtract(igg[:,0:39,:,:].to(device))
            graph_loader = Graph_Dataloader(graph_data, batch_size=1, shuffle=False)
            b = []
            for g_data in graph_loader:
                emb_2 = net.GCNembedding(g_data.to(device))
                b.append(emb_2.data)

            query = torch.concat((emb1,b[-1]), dim=1)
            emb = catEmbeding(query, support)
            n = emb.size(2)
            emb = emb.reshape(-1,n)
            outputs = net.Gx(emb)

        elif net_type==3:
            # Relation Network
            query = net.embeddingExtract(igg.to(device))
            emb = catEmbeding(query, support)
            n = emb.size(2)
            emb = emb.reshape(-1,n)
            outputs = net.Gx(emb)

        elif net_type==4:
            # Matching Network
            query = net(igg.to(device))
            outputs = cosine_score(query, support)

        probability_Hydrothermal = F.softmax(outputs).data.cpu()[0,0].numpy()
        probability_Porphyry = F.softmax(outputs).data.cpu()[0,2].numpy()
        probability_Skarn = F.softmax(outputs).data.cpu()[0,3].numpy()
        probability_Volcano = F.softmax(outputs).data.cpu()[0,4].numpy()

        Hydrothermal_img[h][w] = probability_Hydrothermal
        Porphyry_img[h][w] = probability_Porphyry
        Skarn_img[h][w] = probability_Skarn
        Volcano_img[h][w] = probability_Volcano

iter_val = 0

for i in range(height):

    for j in range(width):
        iter_val = iter_val + 1
        print("processing validation img No "+str(iter_val)+" image")
        graph_data = get_graph_data(graph_dataset=graph_dataset, index=iter_val)
        ig = im[i*pixSize:(i+1)*pixSize, j*pixSize:(j+1)*pixSize]    # 滑动窗口
        create_numpy(ig,i,j,graph_data)

if net_type==1:
    file_name = "ProtoNet"
elif net_type==2:
    file_name = "Geo-Meta"
elif net_type==3:
    file_name = "RelationNet"
elif net_type==4:
    file_name = "MatchingNet"

# Save the numpy heatmap matrix
np.save("../numpyData/{}_Hydrothermal_img.npy".format(file_name), Hydrothermal_img)
np.save("../numpyData/{}_Skarn_img.npy".format(file_name), Skarn_img)
np.save("../numpyData/{}_Porphyry_img.npy".format(file_name), Porphyry_img)
np.save("../numpyData/{}_Volcano_img.npy".format(file_name), Volcano_img)
