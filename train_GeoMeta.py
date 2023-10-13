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
from tools import loadTifImage
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import os

# Calculate the Euclidean distance between two vectors
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    # a is the query set, replicate it m times along the first dimension
    # b is the support set, replicate it n times along the zeroth dimension
    # Calculate the distance between each query image's embedding and the distance to all class prototypes
    # (subtract corresponding positions of vectors, square the result, sum the squared values to get the distance)
    # Negative distance values indicate proximity to class prototypes (smaller negative values indicate closer proximity)
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)

    return logits

# Concatenate the feature maps
def catEmbeding(a, b):
    n = a.shape[0]
    m = b.shape[0]
    emb = torch.cat([a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)], dim=2)
    return emb

# Calculate the accuracy
def accuracy(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum().item() / targets.size(0)

# Calculate the total number of correct predictions
def correct(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum()

# Structure of the feature extractor based on CNN
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

# DDMM
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

# Structure of the feature extractor based on GCN
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

# define the loss function
ce_loss = nn.CrossEntropyLoss()

# After introducing unlabeled data, calculate new prototypes for all categories
def get_new_support(support, unlabel_data, z_j_c, shot, ways):
    prototype = torch.zeros((ways, support.shape[1])).to(support.device)
    m = unlabel_data.shape[0]
    n = unlabel_data.shape[1]

    for i in range(ways):
        prototype[i] = torch.sum(unlabel_data * z_j_c[:,i].unsqueeze(1).expand(m,n), dim=0) +\
                       support[i*shot:i*shot+shot, :].sum(dim=0)
        prototype[i] = prototype[i]/(shot + z_j_c[:,i].sum())
    return prototype

# Given the filename of an image, retrieve the graph data corresponding to that image
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

# Sort the list of filenames according to the specified index
def sort_by_position(lst, positions):
    return [lst[i] for i in positions]

# Split the batch into support set and query set, compute class prototypes from the support set,
# calculate the loss by computing the distances between the query set and the support set.
def fast_adapt(graph_dataset, model, batch, ways, shot, unlabel_shot, query_num, metric=None, device=None):
    if metric is None:
        metric = catEmbeding
    if device is None:
        device = model.device()

    # filenamelist is a list of image names
    data, labels, filenamelist = batch
    data = data.to(device)
    labels = labels.to(device)
    '''
    Data format of a batch
    [Data(x=[100, 3], edge_index=[2, 1368],pos=[100, 2], label=1, filename='392.tif'),
    Data(x=[100, 3], edge_index=[2, 1368], pos=[100, 2], label=1, filename='4065.tif'),
    Data(x=[100, 3], edge_index=[2, 1368], pos=[100, 2], label=4, filename='5581.tif'),
    Data(x=[100, 3], edge_index=[2, 1368], pos=[100, 2], label=4, filename='5900.tif')]
    '''
    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)         # [num_classes * num_shots, c, h, w]
    data = data[:,0:39,:,:]
    labels = labels.squeeze(0)[sort.indices].squeeze(0)     # [num_classes * num_shots]
    filenames = []

    '''
    Retrieve the graph structure data corresponding to each image data within a batch. 
    Use the file names as the key for querying the data.
    '''
    for i in filenamelist:
        filenames.append(os.path.basename(i[0]))
    file_indices = sort.indices.cpu().numpy()
    file_indices = file_indices.squeeze(0)
    file_indices = list(file_indices)
    filenames_sorted = sort_by_position(filenames, file_indices)
    graph_data = get_sorted_graph_data(filenamelist=filenames_sorted, graph_dataset=graph_dataset)

    # Use a CNN-based feature extractor to extract features from geophysical and geochemical data
    CNN_embeddings = model.embeddingExtract(data) # (90,128)

    # Use a GCN-based feature extractor to extract features from geological data
    graph_loader = Graph_Dataloader(graph_data, batch_size=90, shuffle=False, drop_last=False, num_workers=3)
    b = []
    for g_data in graph_loader:
        embeddings_graph = model.GCNembedding(g_data.to(device)) # (90,64)
        b.append(embeddings_graph.data)

    embeddings_integrate = torch.concat((CNN_embeddings, b[-1]), dim=1)  # fuse feature maps

    embeddings = embeddings_integrate

    # Separate support and query images
    support_indices = np.zeros(data.size(0), dtype=bool)  # Indices of labeled data
    unlabel_indices = np.zeros(data.size(0), dtype=bool)  # Indices of unlabeled data
    selection = np.arange(ways) * (shot + unlabel_shot + query_num)

    for offset in range(shot):
        support_indices[selection + offset] = True

    for offset in range(unlabel_shot):
        unlabel_indices[selection + shot + offset] = True

    query_indices = torch.from_numpy(~support_indices + unlabel_indices)  # Indices of the samples in the query set
    support_indices = torch.from_numpy(support_indices)  # Indices of the samples in the support set
    unlabel_indices = torch.from_numpy(unlabel_indices)  # Indices of the unlabeled samples

    # Merge the support set to obtain class prototypes
    support = embeddings[support_indices]
    support_old = support.reshape(ways, shot, -1).sum(dim=1)  # support:[num_classes, embedding]

    unlabel_data = embeddings[unlabel_indices]
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    emb = catEmbeding(unlabel_data, support_old)
    n = emb.size(2)
    emb = emb.reshape(-1, n)
    z_j_c = F.softmax(model.Gx(emb), dim=1)

    support_new = get_new_support(support.detach(), unlabel_data.detach(), z_j_c.detach(), shot, ways)
    save_support = support_new.data

    emb = catEmbeding(query, support_new)
    n = emb.size(2)
    emb = emb.reshape(-1, n)

    logits = model.Gx(emb)

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
            embeddings_integrate = torch.concat((embeddings, a[-1]), dim=1)
            query = embeddings_integrate
            emb = catEmbeding(query, support)
            n = emb.size(2)
            emb = emb.reshape(-1, n)
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

    parser.add_argument('--train_unlabel_shot', type=int, default=5)
    parser.add_argument('--test_unlabel_shot', type=int, default=5)

    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=15)

    parser.add_argument('--test-shot', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--test-query', type=int, default=5)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0')
    print("Use device {}".format(device))

    model = ProtoGCN(args.train_way) # 5 catagories in total
    model.to(device)

    transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset(Need to modify to your own dataset path)
    # Training set
    train_dataset = loadTifImage.DatasetFolder(root='./train/raw/data',
                                               transform=transform)
    # Validation set
    valid_dataset = loadTifImage.DatasetFolder(root='./val/raw/data',
                                               transform=transform)
    # Test set
    test_dataset = loadTifImage.DatasetFolder(root='./val/raw/data',
                                              transform=transform)

    # Load the corresponding graph-based dataset (transferred from image-based dataset by running image2graph.py)
    graph_train_dataset = torch.load('./train/processed/data.pt') # graph-based training data
    graph_valid_dataset = torch.load('./val/processed/data.pt') # graph-based validation data
    graph_test_dataset = torch.load('./val/processed/data.pt') # graph-based test data
    # Divide training and testing tasks
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),    
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=200)


    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=False, drop_last=True, num_workers=3)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),    
        KShots(valid_dataset, args.test_query + args.test_shot), 
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset, task_transforms=valid_transforms, num_tasks=50)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=False, drop_last=True, num_workers=3)

    #  the traditional deep learning testing method
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=90, drop_last=True, num_workers=3)

    # Set up the optimizer
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
    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        temp_acc = 0.0
        temp_support = None

        for i, batch in enumerate(train_loader):
            loss, acc, support = fast_adapt(graph_train_dataset,
                                            model,
                                            batch,
                                            args.train_way,
                                            args.shot,
                                            args.train_unlabel_shot,
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
        
        # Print training progress
        print('epoch {}/{}, train, loss={:.4f} acc={:.4f}'.format(
            epoch, args.max_epoch, n_loss/loss_ctr, n_acc/loss_ctr))
        train_ACC.append(n_acc/loss_ctr)

        # Validation
        model.eval()
        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        for i, batch in enumerate(valid_loader):
            loss, acc, _ = fast_adapt(graph_valid_dataset,
                                      model,
                                      batch,
                                      args.test_way,
                                      args.test_shot,
                                      args.test_unlabel_shot,
                                      args.test_query,
                                      metric=catEmbeding,
                                      device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc
        # Save model weights of this epoch
        torch.save(model.state_dict(), './saved_model/3ProtoGCN_lastepoch.pth')
        torch.save(temp_support.data.cpu(), "./saved_model/3ProtoGCN_lastepoch_support.pt")

        model.eval()
        save_model_para = model.state_dict()
        save_support = temp_support

        temp_support = temp_support.to(device)
        acc_by_deep_learning_way = test_by_deep_learning_way(graph_valid_dataset, temp_support, test_loader, model, device)
        acc_by_deep_learning_way = acc_by_deep_learning_way.item()

        # Save model weights of best epoch
        if acc_by_deep_learning_way > best_acc_by_deep_learning_way:
            torch.save(save_model_para,
                       './saved_model/geometa-ACC_byDL_{:.4f}.pth'.format(acc_by_deep_learning_way))
            torch.save(save_support.data.cpu(),
                       "./saved_model/geometa-support_ACC_byDL_{:.4f}.pt".format(acc_by_deep_learning_way))

            best_support = temp_support
            best_acc_by_deep_learning_way = acc_by_deep_learning_way

        print('epoch {}/{}, val, loss={:.4f} acc={:.4f} acc_by_DL={:.4f}'.format(
            epoch, args.max_epoch, n_loss / loss_ctr, n_acc / loss_ctr, acc_by_deep_learning_way))

        acc = n_acc / loss_ctr

        ACC.append(acc)
        ACC_byDL.append(acc_by_deep_learning_way)
        LOSS.append(n_loss/loss_ctr)
