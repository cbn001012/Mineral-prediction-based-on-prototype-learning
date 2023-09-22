from torch_geometric.data import Dataset, Data
from typing import Callable, List, Optional
import torch
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass
from torch_geometric.loader import DataLoader
import scipy.ndimage

knn_graph = 8  # maximum number of neighbors for each node


def sparsify_graph(A, knn_graph):
    if knn_graph is not None and knn_graph < A.shape[0]:
        idx = np.argsort(A, axis=0)[:-knn_graph, :]  # 取横向距离最小的8个值 也就是距离当前点距离最近的8个点
        np.put_along_axis(A, idx, 0, axis=0)
        idx = np.argsort(A, axis=1)[:, :-knn_graph]  # 纵向
        np.put_along_axis(A, idx, 0, axis=1)
    return A


def spatial_graph(coord, img_size, knn_graph):
    coord = coord / np.array(img_size, float)
    dist = cdist(coord, coord)  # 两组点之间欧式距离
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma ** 2)  # 计算两点之间的距离
    A[np.diag_indices_from(A)] = 0  # 让对角线上的值为零，去除自身
    sparsify_graph(A, knn_graph)
    return A  # 边的邻接矩阵


def visualize_superpixels(avg_values, superpixels):  # 超像素化后可视化
    n_ch = avg_values.shape[1]
    img_sp = np.zeros((*superpixels.shape, n_ch))
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            img_sp[:, :, c][mask] = avg_values[(sp - 1), c]
    return img_sp


def superpixel_features(img, superpixels):
    n_sp = len(np.unique(superpixels))
    n_ch = img.shape[2]
    avg_values = np.zeros((n_sp, n_ch))
    coord = np.zeros((n_sp, 2))
    masks = []
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        for c in range(n_ch):
            avg_values[(sp - 1), c] = np.mean(img[:, :, c][mask])
        coord[(sp - 1)] = np.array(center_of_mass(mask))  # row, col
        masks.append(mask)
    return avg_values, coord, masks


def get_img_matrix(path):
    image_ = io.imread(path)
    image_ = image_[:,:,:]
    img_ = np.array(image_)
    img_ = (img_ / float(img_.max())).astype(np.float32)
    superpixels_ = slic(img_, n_segments=150)  # 超像素分割 为图片增加标签值
    avg_values_, coord_, masks_ = superpixel_features(img_, superpixels_)  # 提取超像素特征
    A_spatial_ = spatial_graph(coord_, img_.shape[:2], knn_graph=8)  # 为每个节点选择旁边的8个节点为邻居
    return coord_, avg_values_, A_spatial_


def get_img_matrix_(img):
    img_ = np.array(img)
    img_ = (img_ / float(img_.max())).astype(np.float32)
    superpixels_ = slic(img_, n_segments=150)  # 超像素分割 为图片增加标签值
    avg_values_, coord_, masks_ = superpixel_features(img_, superpixels_)  # 提取超像素特征
    A_spatial_ = spatial_graph(coord_, img_.shape[:2], knn_graph=8)  # 为每个节点选择旁边的8个节点为邻居
    return coord_, avg_values_, A_spatial_

# 定义自己的数据集类
class mydataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(mydataset, self).__init__(root, transform, pre_transform)

    # 原始文件位置
    @property
    def raw_file_names(self):
        return ['data', 'train', 'val', 'test']

    # 文件保存位置
    @property
    def processed_file_names(self):
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        pass

    # 数据处理逻辑
    def process(self):
        data_list = []
        # 生成邻接矩阵边
        A = []
        B = []
        edge_1 = []
        edge_2 = []
        for i in range(10):
            for j in range(10):
                A.append([i, j])
                B.append([i, j])
        for i in range(100):
            for j in range(100):
                if (A[i][0] - B[j][0]) ** 2 + (A[i][1] - B[j][1]) ** 2 < 2.5:
                    if i != j:
                        edge_1.append(i)
                        edge_1.append(j)
                        edge_2.append(j)
                        edge_2.append(i)
        edge = [edge_1, edge_2]
        label = ['hydrothermal', 'negative',  'porphyry', 'skarn', 'volcano'] # 五分类标签
        path_name = self.raw_paths[0]
        k = 0
        for i in range(5):
            tif_path = os.path.join(path_name, label[i])
            if i == 0:
                tif_list = os.listdir(tif_path)
                for file in tif_list:
                    tif_path_ = os.path.join(tif_path, file)
                    tif_coord, tif_avg_values, tif_A_spatial = get_img_matrix(tif_path_)
                    tif_x = torch.tensor(tif_avg_values, dtype=torch.float32)
                    tif_edge_index = torch.tensor(edge, dtype=torch.long)
                    tif_label = torch.tensor(i, dtype=torch.int64)
                    tif_pos = torch.tensor(tif_coord, dtype=torch.float32)

                    # 定义图结构数据对应的图片数据，以便于进行转化
                    tif_filename = str(file)

                    # tif_edge_attr = torch.tensor(attr)
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
                    # torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(k)))
                    # k += 1
                    data_list.append(data)
            elif i == 1:
                tif_list = os.listdir(tif_path)
                for file in tif_list:
                    tif_path_ = os.path.join(tif_path, file)
                    tif_coord, tif_avg_values, tif_A_spatial = get_img_matrix(tif_path_)
                    tif_x = torch.tensor(tif_avg_values, dtype=torch.float32)
                    tif_edge_index = torch.tensor(edge, dtype=torch.long)
                    tif_label = torch.tensor(i, dtype=torch.int64)
                    tif_pos = torch.tensor(tif_coord, dtype=torch.float32)

                    tif_filename = str(file)

                    # tif_edge_attr = torch.tensor(attr)
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename = tif_filename)
                    # torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(k)))
                    # k += 1
                    data_list.append(data)
            elif i == 2:
                tif_list = os.listdir(tif_path)
                for file in tif_list:
                    tif_path_ = os.path.join(tif_path, file)
                    tif_coord, tif_avg_values, tif_A_spatial = get_img_matrix(tif_path_)
                    tif_x = torch.tensor(tif_avg_values, dtype=torch.float32)
                    tif_edge_index = torch.tensor(edge, dtype=torch.long)
                    tif_label = torch.tensor(i, dtype=torch.int64)
                    tif_pos = torch.tensor(tif_coord, dtype=torch.float32)

                    tif_filename = str(file)
                    # tif_edge_attr = torch.tensor(attr)
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
                    # torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(k)))
                    # k += 1
                    data_list.append(data)
            elif i == 3:
                tif_list = os.listdir(tif_path)
                for file in tif_list:
                    tif_path_ = os.path.join(tif_path, file)
                    tif_coord, tif_avg_values, tif_A_spatial = get_img_matrix(tif_path_)
                    tif_x = torch.tensor(tif_avg_values, dtype=torch.float32)
                    tif_edge_index = torch.tensor(edge, dtype=torch.long)
                    tif_label = torch.tensor(i, dtype=torch.int64)
                    tif_pos = torch.tensor(tif_coord, dtype=torch.float32)

                    tif_filename = str(file)
                    # tif_edge_attr = torch.tensor(attr)
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
                    # torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(k)))
                    # k += 1
                    data_list.append(data)
            elif i == 4:
                tif_list = os.listdir(tif_path)
                for file in tif_list:
                    tif_path_ = os.path.join(tif_path, file)
                    tif_coord, tif_avg_values, tif_A_spatial = get_img_matrix(tif_path_)
                    tif_x = torch.tensor(tif_avg_values, dtype=torch.float32)
                    tif_edge_index = torch.tensor(edge, dtype=torch.long)
                    tif_label = torch.tensor(i, dtype=torch.int64)
                    tif_pos = torch.tensor(tif_coord, dtype=torch.float32)

                    tif_filename = str(file)
                    # tif_edge_attr = torch.tensor(attr)
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
                    # torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(k)))
                    # k += 1
                    data_list.append(data)
        torch.save(data_list, os.path.join(self.processed_dir, f'data.pt'))

    # 定义总数据长度
    def len(self):
        label = ['hydrothermal', 'negative',  'porphyry', 'skarn', 'volcano']
        path_name = self.raw_paths[0]
        tif_path1 = os.path.join(path_name, label[0])
        tif_list1 = os.listdir(tif_path1)
        len1 = len(tif_list1)

        tif_path2 = os.path.join(path_name, label[1])
        tif_list2 = os.listdir(tif_path2)
        len2 = len(tif_list2)

        tif_path3 = os.path.join(path_name, label[2])
        tif_list3 = os.listdir(tif_path3)
        len3 = len(tif_list3)

        tif_path4 = os.path.join(path_name, label[3])
        tif_list4 = os.listdir(tif_path4)
        len4 = len(tif_list4)

        tif_path5 = os.path.join(path_name, label[4])
        tif_list5 = os.listdir(tif_path5)
        len5 = len(tif_list5)
        return len1 + len2 + len3 + len4 + len5

        # 定义获取数据方法

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data

train_dataset = mydataset("") # 指向训练数据（图片）路径
val_dataset = mydataset("")# 指向验证数据（图片）路径
test_dataset = mydataset("")# 指向测试数据（图片）路径

