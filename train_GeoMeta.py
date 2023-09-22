import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as Graph_Dataloader
import torchvision.transforms as transforms
import torchvision as tv
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
import learn2learn as l2l
from lib import loadTifImage
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '3'

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

# 拼接特征图
def catEmbeding(a, b):
    n = a.shape[0]
    m = b.shape[0]

    # a是查询集，将其在第 1维度复制 m份
    # b是支持集，将其在第 0维度复制 n份
    emb = torch.cat([a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)], dim=2)
    return emb

#计算准确率
def accuracy(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum().item() / targets.size(0)

# 计算预测正确的总数
def correct(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum()

# 原型网络的特征提取网络
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

# 动态距离度量模块
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

# 图卷积网络
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
        #self.Gx = Gx(ways)
        self.GCNembedding = GCN()

    def forward(self,):
        pass

# 使用交叉熵损失优化
ce_loss = nn.CrossEntropyLoss()


# 引入无标签数据后，计算新的支持集类原型
def get_new_support(support, unlabel_data, z_j_c, shot, ways):
    prototype = torch.zeros((ways, support.shape[1])).to(support.device)
    m = unlabel_data.shape[0]
    n = unlabel_data.shape[1]

    for i in range(ways):
        prototype[i] = torch.sum(unlabel_data * z_j_c[:,i].unsqueeze(1).expand(m,n), dim=0) +\
                       support[i*shot:i*shot+shot, :].sum(dim=0)
        prototype[i] = prototype[i]/(shot + z_j_c[:,i].sum())
    return prototype

# 给定图片的文件名，取出该图片对应的图数据
def get_graph_data(filenamelist, graph_dataset):
    graph_data = []
    filenames = []
    for i in filenamelist:
        filenames.append(os.path.basename(i[0]))
    for name in filenames:
        for g_data in graph_dataset:
            if g_data.filename == name:
                graph_data.append(g_data)
                continue
    return graph_data

def get_sorted_graph_data(filenamelist, graph_dataset):
    graph_data = []
    for name in filenamelist:
        for g_data in graph_dataset:
            if g_data.filename == name:
                graph_data.append(g_data)
                continue
    return graph_data

def get_graph_data_dl(filenamelist, graph_dataset):
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

# 将文件名列表按照指定的索引排序
def sort_by_position(lst, positions):
    return [lst[i] for i in positions]

# batch为数据、标签和对应的样本绝对路径,ways=30 shot=1 query_num=15
# 将 batch拆分为支持集与查询集，支持集计算出类原型，查询集算出与支持集的距离计算损失
def fast_adapt(graph_dataset, model, batch, ways, shot, unlabel_shot, query_num, metric=None, device=None):
    if metric is None:
        metric = pairwise_distances_logits
    if device is None:
        device = model.device()

    # filenamelist为图片名称
    data, labels, filenamelist = batch
    data = data.to(device)          # [1, 分类数*每类总数, c, h, w]
    labels = labels.to(device)      # [1, 分类数*每类总数]
    '''
    形如
    [Data(x=[100, 3], edge_index=[2, 1368],pos=[100, 2], label=1, filename='392.tif'),
    Data(x=[100, 3], edge_index=[2, 1368], pos=[100, 2], label=1, filename='4065.tif'),
    Data(x=[100, 3], edge_index=[2, 1368], pos=[100, 2], label=4, filename='5581.tif'),
    Data(x=[100, 3], edge_index=[2, 1368], pos=[100, 2], label=4, filename='5900.tif')]
    '''

    # Sort data samples by labels
    # 将数据与标签排序
    sort = torch.sort(labels)       # 按标签从小到大排序
    data = data.squeeze(0)[sort.indices].squeeze(0)         # [分类数*每类总数, c, h, w]
    data = data[:,0:39,:,:]
    labels = labels.squeeze(0)[sort.indices].squeeze(0)     # [分类数*每类总数]
    filenames = []
    for i in filenamelist:
        filenames.append(os.path.basename(i[0]))
    file_indices = sort.indices.cpu().numpy()
    file_indices = file_indices.squeeze(0)
    file_indices = list(file_indices)
    filenames_sorted = sort_by_position(filenames, file_indices)
    graph_data = get_sorted_graph_data(filenamelist=filenames_sorted, graph_dataset=graph_dataset)

    # 原型网络特征提取器提取化探特征
    CNN_embeddings = model.embeddingExtract(data) # (90,128)

    # 图卷积网络提取地质特征
    graph_loader = Graph_Dataloader(graph_data, batch_size=90, shuffle=False, drop_last=False, num_workers=3)
    b = []
    for g_data in graph_loader:
        embeddings_graph = model.GCNembedding(g_data.to(device)) # (90,64)
        b.append(embeddings_graph.data)

    embeddings_integrate = torch.concat((CNN_embeddings, b[-1]), dim=1)  # 将两个特征图进行拼接融合作为最终嵌入 (90, 128+64=192)

    embeddings = embeddings_integrate

        # 将support与 query 的图片分开 support_indices 保存的为支持集的下标
    support_indices = np.zeros(data.size(0), dtype=bool)  # 标签数据的下标
    unlabel_indices = np.zeros(data.size(0), dtype=bool)  # 无监督数据的下标
    selection = np.arange(ways) * (shot + unlabel_shot + query_num)

    for offset in range(shot):
        support_indices[selection + offset] = True

    for offset in range(unlabel_shot):
        unlabel_indices[selection + shot + offset] = True

    query_indices = torch.from_numpy(~support_indices + unlabel_indices)  # 查询集的下标
    support_indices = torch.from_numpy(support_indices)  # 支持集的下标
    unlabel_indices = torch.from_numpy(unlabel_indices)  # 非监督数据的下标

    # 将支持集合并得到类原型
    support = embeddings[support_indices]
    support_old = support.reshape(ways, shot, -1).sum(dim=1)  # support:[类别, embedding]

    unlabel_data = embeddings[unlabel_indices]  # [类别 × 非监督数据数, embedding]
    query = embeddings[query_indices]  # [类别 × 查询数, embedding]
    labels = labels[query_indices].long()
    z_j_c = F.softmax(pairwise_distances_logits(unlabel_data, support_old), dim=1)

    support_new = get_new_support(support.detach(), unlabel_data.detach(), z_j_c.detach(), shot, ways)
    save_support = support_new.data

    logits = pairwise_distances_logits(query, support_new)

    loss_train = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)


    return loss_train, acc, save_support

def test_by_deep_learning_way(graph_dataset,support, test_loader, model, device=None):
    num = 0
    cor_num = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, filenamelist = data
            inputs, labels, filenamelist = inputs[:,0:39,:,:].to(device), labels.to(device), filenamelist
            filenamelist = list(filenamelist)
            embeddings = model.embeddingExtract(inputs)
            graph_data = get_graph_data_dl(filenamelist=filenamelist, graph_dataset=graph_dataset)

            graph_loader = Graph_Dataloader(graph_data, batch_size=90, shuffle=False, num_workers=3)
            a = []
            for g_data in graph_loader:
                embeddings_graph = model.GCNembedding(g_data.to(device))
                a.append(embeddings_graph.data)
            embeddings_integrate = torch.concat((embeddings, a[-1]), dim=1)  # 将两个特征图进行拼接融合作为最终嵌入
            query = embeddings_integrate
            logits = pairwise_distances_logits(query, support)
            cor_num += correct(logits, labels)
            num += inputs.size(0)
    return cor_num/num

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    setup_seed(1012)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)

    parser.add_argument('--train_unlabel_shot', type=int, default=2)  # 非监督数据 2个/类
    parser.add_argument('--test_unlabel_shot', type=int, default=1)  # 非监督数据 1个/类

    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=5)

    parser.add_argument('--test-shot', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-query', type=int, default=5)

    # parser.add_argument('--gpu', default=1)
    args = parser.parse_args()
    print(args)


    # if args.gpu:
    #     device = torch.device('cpu')
    #     print("Use device {}".format(device))
    device = torch.device('cuda:0')
    print("Use device {}".format(device))

    model = ProtoGCN(args.train_way)
    model.to(device)
    model.GCNembedding.load_state_dict(torch.load('./saved_model/GCN_dict.pth'))
    # for param in model.GCNembedding.parameters():
    #     param.requires_grad = False

    transform = transforms.Compose([transforms.ToTensor()])


    # 加载采矿数据集
    train_dataset = loadTifImage.DatasetFolder(root='./train/raw/data',
                                               transform=transform)

    valid_dataset = loadTifImage.DatasetFolder(root='./val/raw/data',
                                               transform=transform)

    test_dataset = loadTifImage.DatasetFolder(root='./val/raw/data',
                                              transform=transform)

    # 加载对应的图数据集
    graph_train_dataset = torch.load('./train/processed/data.pt') # 训练集图数据
    graph_valid_dataset = torch.load('./val/processed/data.pt') # 验证集图数据
    graph_test_dataset = torch.load('./val/processed/data.pt') # 测试集图数据

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),   # 5分类
        KShots(train_dataset, args.train_query + args.shot + args.train_unlabel_shot),    # 15(查询数据)+3(支持集)+2(无监督数据) 张图片
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=200)

    # pytorch code
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=False, drop_last=True, num_workers=3)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),    # 5类
        KShots(valid_dataset, args.test_query + args.test_shot + args.test_unlabel_shot),    # 5+3+1 张图
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset, task_transforms=valid_transforms, num_tasks=50)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=False, drop_last=True, num_workers=3)

    # 深度学习的测试方式
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=90, drop_last=True, num_workers=3)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    best_support = None
    best_acc = 0.0
    best_acc_by_deep_learning_way = 0.0
    ACC = []
    ACC_byDL = []
    LOSS = []
    train_ACC = []


    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        temp_acc = 0.0
        temp_support = None

        # 需要遍历的次数
        # for i in range(400):
        for i, batch in enumerate(train_loader):
            # batch = next(iter(train_loader))    # 5分类 每类 15+2张图片
            loss, acc, support = fast_adapt(graph_train_dataset,
                                            model,
                                            batch,
                                            args.train_way,
                                            args.shot,
                                            args.train_unlabel_shot,
                                            args.train_query,
                                            metric=pairwise_distances_logits,
                                            device=device)

            if acc>temp_acc:
                temp_acc = acc
                temp_support = support

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        print('epoch {}/{}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, args.max_epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        train_ACC.append(n_acc/loss_ctr)

        # 验证
        model.eval()
        # loss = []
        # acc = []
        # acc_by_DL = []

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        # temp_acc = 0.0

        # 200个任务集，会遍历200次
        for i, batch in enumerate(valid_loader):
            loss, acc, _ = fast_adapt(graph_valid_dataset,
                                      model,
                                      batch,
                                      args.test_way,
                                      args.test_shot,
                                      args.test_unlabel_shot,
                                      args.test_query,
                                      metric=pairwise_distances_logits,
                                      device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        torch.save(model.state_dict(), './saved_model/3ProtoGCN_lastepoch.pth')
        torch.save(temp_support.data.cpu(), "./saved_model/3ProtoGCN_lastepoch_support.pt")

        model.eval()
        save_model_para = model.state_dict()
        save_support = temp_support

        temp_support = temp_support.to(device)
        acc_by_deep_learning_way = test_by_deep_learning_way(graph_valid_dataset, temp_support, test_loader, model, device)
        acc_by_deep_learning_way = acc_by_deep_learning_way.item()

        if acc_by_deep_learning_way > best_acc_by_deep_learning_way:
            torch.save(save_model_para,
                       './saved_model/2ProtoGCN-ACC_byDL_{:.4f}.pth'.format(acc_by_deep_learning_way))
            torch.save(save_support.data.cpu(),
                       "./saved_model/2ProtoGCN-support_ACC_byDL_{:.4f}.pt".format(acc_by_deep_learning_way))

            best_support = temp_support
            best_acc_by_deep_learning_way = acc_by_deep_learning_way
        # loss.append(n_loss/loss_ctr)
        # acc.append(n_acc/loss_ctr)
        # acc_by_DL.append()

        print('epoch {}/{}, val, loss={:.4f} acc={:.4f} acc_by_DL={:.4f}'.format(
            epoch, args.max_epoch, n_loss / loss_ctr, n_acc / loss_ctr, acc_by_deep_learning_way))

        acc = n_acc / loss_ctr

        ACC.append(acc)
        ACC_byDL.append(acc_by_deep_learning_way)
        LOSS.append(n_loss/loss_ctr)


'''
原型网络特征提取网络结构
Convnet(
  (encoder): CNN4Backbone(
    (0): ConvBlock(
      (max_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      (normalize): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv): Conv2d(39, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (1): ConvBlock(
      (max_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      (normalize): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (2): ConvBlock(
      (max_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      (normalize): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    (3): ConvBlock(
      (max_pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
      (normalize): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU()
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
  )
)'''