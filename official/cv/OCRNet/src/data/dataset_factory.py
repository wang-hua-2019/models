import os
import cv2
import multiprocessing
from mindspore import dataset as ds
from .cityscapes import Cityscapes
from .transforms_factory import create_transform


class BulidingDataset:
    """
    Pass in a custom dataset that conforms to the format.

    Args:

        dataset_root (str): The dataset directory.

        Examples:
            dataset_root = 'dataset_root_path'
            dataset = Dataset(dataset_root = dataset_root,
                              mode = 'train')

    """

    def __init__(self,
                 dataset_root,
                 mode='train'):
        self.dataset_root = dataset_root
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode

        if mode.lower() not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        self.dataset_root = dataset_root
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))

        # TODO 请根据实际文件存放方式编写下述代码
        if mode == 'train':
            image_dir = os.path.join(self.dataset_root, 'img_train')
        elif mode == 'val':
            image_dir = os.path.join(self.dataset_root, 'img_val')
        else:
            raise NotImplementedError

        image_paths = os.walk(image_dir)

        for path, dir_lst, file_lst in image_paths:
            for file_name in file_lst:
                if mode == 'train':
                    image_path = os.path.join(self.dataset_root, 'img_train', file_name)
                    label_path = os.path.join(self.dataset_root, 'label_train', file_name)
                elif mode == 'val':
                    image_path = os.path.join(self.dataset_root, 'img_val', file_name)
                    label_path = os.path.join(self.dataset_root, 'label_val', file_name)
                else:
                    raise NotImplementedError
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        return image_path, label_path

    def __len__(self):
        return len(self.file_list)

def create_dataset(cfg, batch_size, num_parallel_workers=8, group_size=1, rank=0, task="train"):
    """
    Creates dataset by name.
    Args:
        cfg (obj): Configs
        batch_size (int): The number of rows each batch is created with.
        num_parallel_workers (int): Number of workers(threads) to process the dataset in parallel.
        group_size (int): Number of shards that the dataset will be divided
        rank (int): The shard ID within `group_size`
        is_train (bool): whether is training.
        task (str): 'train', 'eval' or 'infer'.
    """
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(2)
    ds.config.set_enable_shared_mem(True)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(num_parallel_workers, cores // group_size)
    ds.config.set_num_parallel_workers(num_parallel_workers)
    is_train = task == "train"
    if task == "train":
        trans_config = getattr(cfg, "train_transforms", cfg)
    elif task in ("val", "eval"):
        trans_config = getattr(cfg, "eval_transforms", cfg)
    else:
        raise NotImplementedError
    item_transforms = getattr(trans_config, "item_transforms", [])
    transforms_name_list = []
    for transform in item_transforms:
        transforms_name_list.extend(transform.keys())
    transforms_list = []
    for i, transform_name in enumerate(transforms_name_list):
        transform = create_transform(item_transforms[i])
        transforms_list.append(transform)
    dataset = None
    if cfg.name == "cityscapes":
        dataset = Cityscapes(ignore_label=cfg.ignore_label).create_dataset(
            dataset_dir=cfg.dataset_dir,
            map_label=cfg.map_label,
            group_size=group_size,
            rank=rank,
            is_train=is_train,
        )
    elif cfg.name == "building_dataset":
        dataset = BulidingDataset(dataset_root=cfg.dataset_dir, mode=task)
    else:
        raise NotImplementedError

    dataset = dataset.map(
        operations=transforms_list, input_columns=["image", "label"], python_multiprocessing=True, max_rowsize=64
    )
    dataset = dataset.project(["image", "label"])

    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset, dataset.get_dataset_size()


def get_dataset(cfg):
    if cfg.name == "cityscapes":
        dataset = Cityscapes(ignore_label=cfg.ignore_label)
    elif cfg.name == "building_dataset":
        dataset = BulidingDataset(dataset_root=cfg.dataset_dir)
    else:
        raise NotImplementedError
    return dataset
