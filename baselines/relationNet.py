import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision as tv
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels
from lib import loadTifImage
import os


def catEmbeding(a, b):
    n = a.shape[0]
    m = b.shape[0]

    # a是查询集，将其在第 1维度复制 m份
    # b是支持集，将其在第 0维度复制 n份
    emb = torch.cat([a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)], dim=2)
    return emb


def accuracy(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum().item() / targets.size(0)

def correct(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum()

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
        self.linear1 = nn.Linear(128*2, 512)
        self.linear2 = nn.Linear(512, 1)
        self.ways = ways
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        # x = self.linear3(x)
        return x.reshape(-1, self.ways)

class relationNet(nn.Module):
    def __init__(self, ways):
        super().__init__()
        self.embeddingExtract = Convnet()
        self.Gx = Gx(ways)

    def forward(self,):
        pass

# loss = nn.MSELoss(reduction="sum")
ce_loss = nn.CrossEntropyLoss()

# batch就是数据与标签 ways=30 shot=1 query_num=15
# 将 batch拆分为支持集与查询集，支持集计算出类原型，查询集算出与支持集的距离计算损失
def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric=catEmbeding
    if device is None:
        device = model.device()

    data, labels = batch
    data = data.to(device)          # [1, 分类数*每类总数, c, h, w]
    labels = labels.to(device)      # [1, 分类数*每类总数]

    # Sort data samples by labels
    # 将数据与标签排序
    sort = torch.sort(labels)       # 按标签从小到大排序
    data = data.squeeze(0)[sort.indices].squeeze(0)         # [分类数*每类总数, c, h, w]
    labels = labels.squeeze(0)[sort.indices].squeeze(0)     # [分类数*每类总数]

    # Compute support and query embeddings
    embeddings = model.embeddingExtract(data)

    # 将support与 query 的图片分开 support_indices 保存的为支持集的下标
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True

    query_indices = torch.from_numpy(~support_indices)      # 查询集的下标
    support_indices = torch.from_numpy(support_indices)     # 支持集的下标

    # 将支持集合并得到类原型
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).sum(dim=1)   # support:[类别, embedding]
    save_support = support.data

    query = embeddings[query_indices]                       # [类别 × 查询数, embedding]
    labels = labels[query_indices].long()
    # print('labels1',labels.size())

    emb = catEmbeding(query, support)
    # print('emb1',emb.size())

    n = emb.size(2)
    emb = emb.reshape(-1,n)


    logits = model.Gx(emb)
    # print(logits.size())

    loss_train = ce_loss(logits, labels)
    acc = accuracy(logits, labels)

    return loss_train, acc, save_support

def test_by_deep_learning_way(support, test_loader, model, device=None):
    num = 0
    cor_num = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            query = model.embeddingExtract(inputs)
            emb = catEmbeding(query, support)
            n = emb.size(2)
            emb = emb.reshape(-1,n)
            logits = model.Gx(emb)

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
    setup_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)

    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=15)

    parser.add_argument('--test-shot', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-query', type=int, default=5)

    parser.add_argument('--gpu', default=1)
    args = parser.parse_args()
    print(args)


    device = torch.device('cpu')
    if args.gpu and torch.cuda.device_count():
        print("Using gpu")
        device = torch.device('cuda:1')

    model = relationNet(args.train_way)
    model.to(device)

    # 加载采矿数据集
    train_dataset = loadTifImage.DatasetFolder(root='./tif_data/img/dim39/train',
                                               transform=transforms.ToTensor())

    valid_dataset = loadTifImage.DatasetFolder(root='./tif_data/img/dim39/val',
                                               transform=transforms.ToTensor())

    test_dataset = loadTifImage.DatasetFolder(root='./tif_data/img/dim39/val',
                                              transform=transforms.ToTensor())

    # train_dataset = tv.datasets.MNIST(
    #     root=path_data, train=True, transform=transforms.ToTensor(), download=True)
    # valid_dataset = tv.datasets.MNIST(
    #     root=path_data, train=False, transform=transforms.ToTensor(), download=True)


    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),   # 5分类
        KShots(train_dataset, args.train_query + args.shot),    # 15+2 张图片
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms,
                                       num_tasks=200)

    # pytorch code
    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),    # 5类
        KShots(valid_dataset, args.test_query + args.test_shot),    # 5+2 张图
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=50)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    # 深度学习的测试方式
    test_loader = DataLoader(test_dataset, shuffle=False)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    best_support = None
    best_acc = 0.0
    best_acc_by_deep_learning_way = 0.0
    ACC = []
    ACC_byDL = []



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
            loss, acc, support = fast_adapt(model,
                                            batch,
                                            args.train_way,
                                            args.shot,
                                            args.train_query,
                                            metric=catEmbeding,
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

        # 验证
        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        # temp_acc = 0.0

        # 200个任务集，会遍历200次
        for i, batch in enumerate(valid_loader):
            loss, acc, _ = fast_adapt(model,
                                      batch,
                                      args.test_way,
                                      args.test_shot,
                                      args.test_query,
                                      metric=catEmbeding,
                                      device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

            # if acc>temp_acc:
            #     temp_acc = acc
            #     best_support = support

        model.eval()
        save_model_para = model.state_dict()
        save_support = temp_support

        temp_support = temp_support.to(device)
        acc_by_deep_learning_way = test_by_deep_learning_way(temp_support, test_loader, model, device)
        acc_by_deep_learning_way = acc_by_deep_learning_way.item()

        if acc_by_deep_learning_way>best_acc_by_deep_learning_way:
            torch.save(save_model_para, './saved_model/relationNet-ACC_byDL_{:.4f}.pth'.format(acc_by_deep_learning_way))
            torch.save(save_support.data.cpu(), "./saved_model/relationNet-support_ACC_byDL_{:.4f}.pt".format(acc_by_deep_learning_way))

            best_support = temp_support
            best_acc_by_deep_learning_way = acc_by_deep_learning_way

        print('epoch {}/{}, val, loss={:.4f} acc={:.4f} acc_by_DL={:.4f}'.format(
            epoch, args.max_epoch, n_loss/loss_ctr, n_acc/loss_ctr, acc_by_deep_learning_way))

        acc = n_acc/loss_ctr


        ACC.append(acc)
        ACC_byDL.append(acc_by_deep_learning_way)

        # if n_acc/loss_ctr > best_acc:
        #     best_acc = n_acc/loss_ctr
        #     torch.save(model.state_dict(), './saved_model/CNN4_ACC_{:.4f}.pth'.format(best_acc))

    # np.save("./saved_model/relationNet-trainSetSupport_ACC.npy", ACC)
    # np.save("./saved_model/relationNet-trainSetSupport_ACC_byDL.npy", ACC_byDL)