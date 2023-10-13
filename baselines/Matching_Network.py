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
from tools import loadTifImage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# calculate the cosine similarity between two vectors
def cosine_score(a, b):
    n = a.shape[0]
    m = b.shape[0]

    # a is the query set, copy it m times along the first dimension
    # b is the support set, copy it n times along the zeroth dimension
    cos_score = torch.cosine_similarity(a.unsqueeze(1).expand(n, m, -1),
                                        b.unsqueeze(0).expand(n, m, -1), dim=2)

    return cos_score


def accuracy(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum().item() / targets.size(0)

def correct(predictions, targets):
    _, predicted = torch.max(predictions.data,1)
    return (predicted == targets).sum()

# Define feature extraction network
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

ce_loss = nn.CrossEntropyLoss() # define the loss function

# Split the batch into support set and query set, compute class prototypes from the support set, 
# calculate the loss by computing the distances between the query set and the support set.
def fast_adapt(model, batch, ways, shot, query_num, metric=None, device=None):
    if metric is None:
        metric=cosine_score  # Choose cosine similarity as the distance metric.
    if device is None:
        device = model.device()

    data, labels = batch
    data = data.to(device)           # [1, num_classes * num_shots, c, h, w]
    labels = labels.to(device)       # [1, num_classes * num_shots]

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)         # [num_classes * num_shots, c, h, w]
    labels = labels.squeeze(0)[sort.indices].squeeze(0)     # [num_classes * num_shots]

    # Compute embeddings using backbone
    embeddings = model(data)

    # Separate support and query images
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(ways) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True

    query_indices = torch.from_numpy(~support_indices)      # Indices of query set
    support_indices = torch.from_numpy(support_indices)     # Indices of support set

    # Merge the support set to obtain class prototypes
    support = embeddings[support_indices]
    support = support.reshape(ways, shot, -1).sum(dim=1)   # support:[num_classes, embedding]
    save_support = support.data

    query = embeddings[query_indices]                      # [num_classes * query_num, embedding]
    labels = labels[query_indices].long()

    cosine_logits = cosine_score(query, support)

    loss_train = ce_loss(cosine_logits, labels)
    acc = accuracy(cosine_logits, labels)

    return loss_train, acc, save_support

def test_by_deep_learning_way(support, test_loader, model, device=None):
    num = 0
    cor_num = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            query = model(inputs)
            cosine_logits = cosine_score(query, support)

            cor_num += correct(cosine_logits, labels)
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
        device = torch.device('cuda')

    model = Convnet()
    model.to(device)

    # load the dataset(Need to modify to your own dataset path)
    # Training set
    train_dataset = loadTifImage.DatasetFolder(root='./tif_data/img/dim39/train',
                                               transform=transforms.ToTensor())
    # Validation set
    valid_dataset = loadTifImage.DatasetFolder(root='./tif_data/img/dim39/val',
                                               transform=transforms.ToTensor())
    # Test set
    test_dataset = loadTifImage.DatasetFolder(root='./tif_data/img/dim39/val',
                                              transform=transforms.ToTensor())

    # Divide training and testing tasks
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        NWays(train_dataset, args.train_way),
        KShots(train_dataset, args.train_query + args.shot),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms,
                                       num_tasks=200)


    train_loader = DataLoader(train_tasks, pin_memory=True, shuffle=True)

    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        NWays(valid_dataset, args.test_way),    
        KShots(valid_dataset, args.test_query + args.test_shot),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=50)
    valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

    #  the traditional deep learning testing method
    test_loader = DataLoader(test_dataset, shuffle=False)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    best_support = None
    best_acc = 0.0
    best_acc_by_deep_learning_way = 0.0
    ACC = []
    ACC_byDL = []

    # Training loop
    for epoch in range(1, args.max_epoch + 1):
        model.train()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0
        temp_acc = 0.0
        temp_support = None

        for i, batch in enumerate(train_loader):
            loss, acc, support = fast_adapt(model,
                                            batch,
                                            args.train_way,
                                            args.shot,
                                            args.train_query,
                                            metric=cosine_score,
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

        # Validation
        model.eval()

        loss_ctr = 0
        n_loss = 0
        n_acc = 0

        # There are 200 task sets, and it will iterate 200 times.
        for i, batch in enumerate(valid_loader):
            loss, acc, _ = fast_adapt(model,
                                      batch,
                                      args.test_way,
                                      args.test_shot,
                                      args.test_query,
                                      metric=cosine_score,
                                      device=device)

            loss_ctr += 1
            n_loss += loss.item()
            n_acc += acc

        model.eval()
        save_model_para = model.state_dict()
        save_support = temp_support

        temp_support = temp_support.to(device)
        acc_by_deep_learning_way = test_by_deep_learning_way(temp_support, test_loader, model, device)
        acc_by_deep_learning_way = acc_by_deep_learning_way.item()
        
        # Save model weights
        if acc_by_deep_learning_way>best_acc_by_deep_learning_way:
            torch.save(save_model_para, './saved_model/matchingNetwork-ACC_byDL_{:.4f}.pth'.format(acc_by_deep_learning_way))
            torch.save(save_support.data.cpu(), "./saved_model/matchingNetwork-support_ACC_byDL_{:.4f}.pt".format(acc_by_deep_learning_way))

            best_support = temp_support
            best_acc_by_deep_learning_way = acc_by_deep_learning_way

        print('epoch {}/{}, val, loss={:.4f} acc={:.4f} acc_by_DL={:.4f}'.format(
            epoch, args.max_epoch, n_loss/loss_ctr, n_acc/loss_ctr, acc_by_deep_learning_way))

        acc = n_acc/loss_ctr


        ACC.append(acc)
        ACC_byDL.append(acc_by_deep_learning_way)
