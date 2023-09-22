import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm
# from model import resnet34
import sys
import learn2learn as l2l
from sklearn.metrics import *  # pip install scikit-learn
from scipy import interp
from lib import loadTifImage
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import os
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
        self.linear1 = nn.Linear(192 * 2, 512)
        self.linear2 = nn.Linear(512, 1)
        self.ways = ways
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x.reshape(-1, self.ways)

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

# class relationNet(nn.Module):
#     def __init__(self, ways):
#         super().__init__()
#         self.embeddingExtract = Convnet()
#         self.Gx = Gx(ways)
#
#     def forward(self,):
#         pass

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

    # a是查询集，将其在第 1维度复制 m份
    # b是支持集，将其在第 0维度复制 n份
    # 算出每一个查询集的图片embedding距离所有类原型的距离 (向量对应位置相减再平方，再求和，得到距离)
    # 距离为负数，值越大表示离类原型越近（负得越小）
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)

    return logits

def catEmbeding(a, b):
    n = a.shape[0]
    m = b.shape[0]

    # a是查询集，将其在第 1维度复制 m份
    # b是支持集，将其在第 0维度复制 n份
    emb = torch.cat([a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)], dim=2)
    return emb

def cosine_score(a, b):
    n = a.shape[0]
    m = b.shape[0]
    # a是查询集，将其在第 1维度复制 m份
    # b是支持集，将其在第 0维度复制 n份
    cos_score = torch.cosine_similarity(a.unsqueeze(1).expand(n, m, -1),
                                        b.unsqueeze(0).expand(n, m, -1), dim=2)
    return cos_score

def get_graph_data(filenamelist, graph_dataset):
    graph_data = []
    filenames = []
    for i in filenamelist:
        filenames.append(os.path.basename(i))
    for name in filenames:
        for g_data in graph_dataset:
            if g_data.filename == name:
                graph_data.append(g_data)
                continue
    return graph_data

# read class_indict
json_path = 'class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

json_file = open(json_path, "r")
class_indict = json.load(json_file)

num_class = 5   # 分类数

# 首先定义一个 分类数*分类数 的空混淆矩阵
conf_matrix = np.zeros((num_class, num_class))

def drawROC(score_list,label_list):
    score_array = np.array(score_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)

    # print("score_array:", score_array.shape)  # (batchsize, classnum)
    # print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    # 调用sklearn库，计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    # 绘制所有类别平均的roc曲线
    plt.figure(figsize=(8,6), dpi=100)
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'gold', 'green'])
    for i, color in zip([0,2,3,4,1], colors):
        mine_name = ''
        if i==0:
            mine_name = 'Hydrothermal'
        elif i==1:
            mine_name = 'No mine'
        elif i==2:
            mine_name = 'Porphyry'
        elif i==3:
            mine_name = 'Skarn'
        else:
            mine_name = 'Volcano'

        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(mine_name, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ProtoGCN')
    plt.legend(loc="lower right")
    # plt.savefig('set113_roc.jpg')
    plt.show(block=True)

def draw_confusion_matrix():

    labels = ['hydrothermal', 'no-mine', 'porphyry', 'skarn', 'volcano']#每种类别的标签

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2	#数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(num_class):
        for y in range(num_class):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()#保证图不重叠
    plt.yticks(range(num_class), labels)
    plt.xticks(range(num_class), labels)#X轴字体倾斜45°
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('c-10 Confusion matrix')
    plt.show(block=True)

def confusion_matrix(labels, preds):
    # print(labels)
    # print(preds)
    for p, t in zip(labels, preds):
        conf_matrix[p, t] += 1

    # 超参数设置
device = torch.device("cuda:2")
print("using {} device.".format(device))

batch_size = 90
nw = 3

# 加载图测试集
graph_test_dataset = torch.load('./processed/data.pt')

# 加载图片测试集
validate_dataset = loadTifImage.DatasetFolder(root='./val/raw/data', transform=transforms.ToTensor())

val_num = len(validate_dataset)

validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=90, shuffle=False, drop_last=True, num_workers = 6)

print("using {} images for validation.".format(val_num))

# 1:protoNet;
# 2:ss-protoNet
# 3:relationNet ;
# 4:marchNet ;
net_type = 3
#9580
model_weight_path = "./saved_model/ProtoGCN-ACC_byDL_0.9657.pth"
support = torch.load('./saved_model/ProtoGCN-support_ACC_byDL_0.9657.pt')
support = support.to(device)

if net_type == 3:
    # 3. relationNet 模型加载
    # net = relationNet(ways=5)       # 5分类
    net = ProtoGCN(ways=5)
else:
    # 1 or 2 or 4. protoNet/ss-protoNet/marchNet 模型加载
    net = Convnet()

assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path))

# 装入GPU
net.to(device)
# net.GCNembedding.load_state_dict(torch.load('/data/Chan/cbn/saved_model/GCN_dict.pth'))
# for param in net.GCNembedding.parameters():
#     param.requires_grad = False

score_list = []  # 存储预测得分
label_list = []  # 存储真实标签

true_label = []
predict_label = []

# validate
net.eval()  # 关闭 dropout
acc = 0.0  # accumulate accurate number / epoch
with torch.no_grad():
    for i, val_data in enumerate(validate_loader):
        val_images, val_labels, val_filenamelist = val_data
        val_filenamelist = list(val_filenamelist)
        val_images = val_images[:, 0:39, :, :]
        true_label.extend(val_labels.data.cpu().numpy())

        if net_type == 1 or net_type == 2:
            # 1 or 2. protoNet/ss-protoNet
            query = net(val_images.to(device))
            outputs = pairwise_distances_logits(query, support)  # 得到logits
        elif net_type == 3:
            # 3. ProtoGCN
            graph_data = get_graph_data(filenamelist=val_filenamelist, graph_dataset=graph_test_dataset)
            graph_loader = Graph_Dataloader(graph_data, batch_size=90, shuffle=False, drop_last=True,num_workers=nw)
            huatan_embedding = net.embeddingExtract(val_images.to(device))
            a = []
            for g_data in graph_loader:
                dizhi_embedding = net.GCNembedding(g_data.to(device))
                a.append(dizhi_embedding.data)
            embeddings_integrate = torch.concat((huatan_embedding, a[-1]), dim=1)
            query = embeddings_integrate
            # query = huatan_embedding

            # query = net.embeddingExtract(val_images.to(device))

            emb = catEmbeding(query, support)
            n = emb.size(2)
            emb = emb.reshape(-1, n)
            outputs = net.Gx(emb)

        elif net_type == 4:
            query = net.embeddingExtract(val_images.to(device))
            emb = catEmbeding(query , support)
            n = emb.size(2)
            emb = emb.reshape(-1,n)
            outputs = net.Gx(emb)


        # ROC曲线数据
        score_tmp = torch.nn.functional.softmax(outputs, dim=1)  # (batchsize, nclass)
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(val_labels.cpu().numpy())

        predict_y = torch.max(outputs, dim=1)[1]
        predict_label.extend(predict_y.data.cpu().numpy())
        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        # 记录混淆矩阵参数
        confusion_matrix(val_labels.cpu().numpy(), predict_y.cpu().numpy())

        # 输出详细信息
        outputs_prob = F.softmax(outputs.cpu(), dim=1).gather(1, val_labels.unsqueeze(-1).cpu()).squeeze(-1)

        outputs_prob = np.array(outputs_prob.data.cpu())
        outputs_label = np.array(predict_y.data.cpu())
        val_labels_cpu = np.array(val_labels.data.cpu())

        cor_or_err = np.equal(outputs_label, val_labels_cpu)

drawROC(score_list, label_list)  # 绘制ROC曲线

val_accurate = acc / val_num
print('val_accuracy: %.4f' % (val_accurate))

print('Finished Testing')

# 精度，准确率， 预测正确的占所有样本种的比例
accuracy = accuracy_score(true_label, predict_label)
print("精度: ", accuracy)

# 查准率P（准确率），precision(查准率)=TP/(TP+FP)

precision = precision_score(true_label, predict_label, labels=None, pos_label=1,
                            average='macro')  # 'micro', 'macro', 'weighted'
print("查准率P: ", precision)

# 查全率R（召回率），原本为对的，预测正确的比例；recall(查全率)=TP/(TP+FN)
recall = recall_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
print("召回率: ", recall)

# F1-Score
f1 = f1_score(true_label, predict_label, average='macro')  # 'micro', 'macro', 'weighted'
print("F1 Score: ", f1)
