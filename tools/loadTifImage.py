import os
import numpy as np
import sys
from torch.utils.data import Dataset
from skimage import transform,io

# 支持的图片格式
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def has_file_allowed_extension(filename, extensions):
    """
    Args:
        filename (string)
        extensions (iterable of strings)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, class_to_idx, extensions):
    """
        Return in the form [(image path, corresponding class index), (), ...].
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def loadTifImage(path):
    image = io.imread(path)
    image = transform.resize(image, (20, 20))
    image = image/255.0
    im = np.array(image, dtype=np.float32)
    return im

class DatasetFolder(Dataset):
    """
     Args:
        root (string): Root directory path.
        loader (callable): A callable function to load a sample given its path.
        extensions (list[string]): A list of acceptable file extensions for images.
        transform (callable, optional): A function that applies transformations to the samples and returns the transformed version.
        E.g., transforms.RandomCrop for images.
        target_transform (callable, optional): A function that applies transformations to the sample labels.

     Attributes:
        classes (list): List of class names.
        class_to_idx (dict): A dictionary mapping class names to class indices, e.g., {'cat': 0, 'dog': 1}.
        samples (list): List of (sample path, class index) tuples, where each tuple represents the path of a sample and its corresponding class index.
        targets (list): List of class indices for each image in the dataset.
    """

    def __init__(self, root, loader=loadTifImage, extensions=IMG_EXTENSIONS, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
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
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
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