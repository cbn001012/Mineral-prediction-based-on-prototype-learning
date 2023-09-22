import os
import numpy as np
import sys
from torch.utils.data import Dataset
from skimage import transform,io

# 支持的图片格式
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def has_file_allowed_extension(filename, extensions):
    """查看文件是否是支持的可扩展类型

    Args:
        filename (string): 文件路径
        extensions (iterable of strings): 可扩展类型列表，即能接受的图像文件类型

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions) # 返回True或False列表

def make_dataset(dir, class_to_idx, extensions):
    """
        返回形如[(图像路径, 该图像对应的类别索引值),(),...]
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): #层层遍历文件夹，返回当前文件夹路径，存在的所有文件夹名，存在的所有文件名
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions): #查看文件是否是支持的可扩展类型，是则继续
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def loadTifImage(path):
    image = io.imread(path)
    # print('image.shape=>',image.shape)
    image = transform.resize(image, (20, 20))     # 修改尺寸，仅能在此处修改 原224*224
    image = image/255.0             # 归一化
    # print(image)
    im = np.array(image, dtype=np.float32)
    return im

class DatasetFolder(Dataset):
    """
     Args:
        root (string): 根目录路径
        loader (callable): 根据给定的路径来加载样本的可调用函数
        extensions (list[string]): 可扩展类型列表，即能接受的图像文件类型.
        transform (callable, optional): 用于样本的transform函数，然后返回样本transform后的版本
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): 用于样本标签的transform函数

     Attributes:
        classes (list): 类别名列表
        class_to_idx (dict): 项目(class_name, class_index)字典,如{'cat': 0, 'dog': 1}
        samples (list): (sample path, class_index) 元组列表，即(样本路径, 类别索引)
        targets (list): 在数据集中每张图片的类索引值，为列表
    """

    def __init__(self, root, loader=loadTifImage, extensions=IMG_EXTENSIONS, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)    # 得到类名和类索引，如['cat', 'dog']和{'cat': 0, 'dog': 1}
        # 返回形如[(图像路径, 该图像对应的类别索引值),(),...]，即对每个图像进行标记
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                           "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]  # 所有图像的类索引值组成的列表

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        在数据集中查找类文件夹。

        Args:
            dir (string): 根目录路径

        Returns:
            返回元组: (classes, class_to_idx)即(类名, 类索引)，其中classes即相应的目录名，如['cat', 'dog'];class_to_idx为形如{类名:类索引}的字典，如{'cat': 0, 'dog': 1}.

        Ensures:
            保证没有类名是另一个类目录的子目录
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]   # 获得根目录dir的所有第一层子目录名
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]   # 效果和上面的一样，只是版本不同方法不同
        classes.sort() #然后对类名进行排序
        class_to_idx = {classes[i]: i for i in range(len(classes))}     # 然后将类名和索引值一一对应的到相应字典，如{'cat': 0, 'dog': 1}
        return classes, class_to_idx    # 然后返回类名和类索引

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)  # 加载图片函数，可自定义为opencv，默认为PIL
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str