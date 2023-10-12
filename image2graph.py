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
    """
        Sparsifies the graph represented by the adjacency matrix A by keeping only the k-nearest neighbors for each node.

        Args:
            A (numpy.ndarray): Adjacency matrix of the graph.
            knn_graph (int): Maximum number of neighbors to keep for each node.

        Returns:
            numpy.ndarray: Sparsified adjacency matrix.
        """
    if knn_graph is not None and knn_graph < A.shape[0]:
        idx = np.argsort(A, axis=0)[:-knn_graph, :]  # Take the 8 smallest values horizontally, which means the 8 points closest to the current point in terms of distance
        np.put_along_axis(A, idx, 0, axis=0)
        idx = np.argsort(A, axis=1)[:, :-knn_graph]  # Vertically
        np.put_along_axis(A, idx, 0, axis=1)
    return A

def spatial_graph(coord, img_size, knn_graph):
    """
        Constructs a spatial graph based on the coordinates of the nodes.

        Args:
            coord (numpy.ndarray): Node coordinates.
            img_size (tuple): Size of the image.
            knn_graph (int): Maximum number of neighbors to consider for each node.

        Returns:
            numpy.ndarray: Adjacency matrix of the spatial graph.
        """
    coord = coord / np.array(img_size, float)
    dist = cdist(coord, coord)  # Euclidean distance between two sets of points
    sigma = 0.1 * np.pi
    A = np.exp(- dist / sigma ** 2)  # Calculate the distance between two points
    A[np.diag_indices_from(A)] = 0  # Set the values on the diagonal to zero, excluding the self
    sparsify_graph(A, knn_graph)
    return A  # Adjacency matrix of the graph

def superpixel_features(img, superpixels):
    """
        Extracts features from superpixels in an image.

        Args:
            img (numpy.ndarray): Input image.
            superpixels (numpy.ndarray): Superpixel segmentation.

        Returns:
            numpy.ndarray: Average values of each channel for each superpixel.
            numpy.ndarray: Coordinates of each superpixel.
            list: List of masks for each superpixel.
        """
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
    """
        Loads and processes an image into a graph data representation.

        Args:
            path (str): Path to the image file.

        Returns:
            numpy.ndarray: Node coordinates.
            numpy.ndarray: Superpixel average values.
            numpy.ndarray: Adjacency matrix of the spatial graph.
        """
    image_ = io.imread(path)
    image_ = image_[:,:,:]
    img_ = np.array(image_)
    img_ = (img_ / float(img_.max())).astype(np.float32)
    superpixels_ = slic(img_, n_segments=150)  # Superpixel segmentation, adding label values to the image
    avg_values_, coord_, masks_ = superpixel_features(img_, superpixels_)  # Extract superpixel features
    A_spatial_ = spatial_graph(coord_, img_.shape[:2], knn_graph=8)  # Select the 8 neighboring nodes for each node
    return coord_, avg_values_, A_spatial_

# Define your own dataset class
class mydataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(mydataset, self).__init__(root, transform, pre_transform)

    # Location of the original files
    @property
    def raw_file_names(self):
        return ['data', 'train', 'val', 'test']

    # File saving location
    @property
    def processed_file_names(self):
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self):
        pass

    # Data processing logic
    def process(self):
        """
            Processes the dataset by converting image data to graph data representation.

            Returns:
                None
        """
        data_list = []
        # Generate edges in the graph based on the adjacency matrix
        A = []
        B = []
        edge_1 = []
        edge_2 = []
        # Each sample in the experimental dataset is a 10x10 pixel image. Adjust the sample size based on the size of samples in your own dataset.
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
        '''
        There are five categories in the experimental dataset, corresponding to the range of i from 0 to 4. 
        Please adjust the loop hyperparameters and set the category labels according to your own situation.
        '''
        label = ['hydrothermal', 'negative',  'porphyry', 'skarn', 'volcano'] # Labels for the five categories
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
                    tif_filename = str(file)
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
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
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename = tif_filename)
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
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
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
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
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
                    data = Data(x=tif_x, edge_index=tif_edge_index, label=tif_label, pos=tif_pos, filename=tif_filename)
                    data_list.append(data)
        torch.save(data_list, os.path.join(self.processed_dir, f'data.pt'))

    # Define the length of the total data
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

        # Define the method to retrieve the data

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data

train_dataset = mydataset("")  # Point to the training data (images) path
val_dataset = mydataset("")    # Point to the validation data (images) path
test_dataset = mydataset("")   # Point to the test data (images) path

