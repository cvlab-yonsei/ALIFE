import torch
from torch.utils.data import DataLoader

from .transform import (
    Compose, 
    RandomResizedCrop, 
    RandomHorizontalFlip, 
    Resize,
    CenterCrop, 
    ToTensor,
    Normalize 
)
from datasets.voc import VOCSegmentationIncremental, VOC_CLASSES
from datasets.ade import AdeSegmentationIncremental, ADE_CLASSES
from utils.distributed_sampler import TrainingSampler, InferenceSampler, trivial_batch_collator, worker_init_reset_seed
import utils.tasks as tasks

__all__ = ["get_datasets"]

def get_datasets(cfg):
    tr_transform = Compose([
        RandomResizedCrop(cfg.DATA.CROP_SIZE, (0.5, 2.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = Compose([
        Resize(size=cfg.DATA.CROP_SIZE),
        CenterCrop(size=cfg.DATA.CROP_SIZE),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    labels, labels_old = tasks.get_task_labels(cfg.dataset, cfg.TASK, cfg.STEP)
    labels_cum = labels_old + labels

    idxs_path = f"./datasets/{cfg.dataset}/{cfg.TASK}"
    if cfg.OVERLAP:
        idxs_path += "-ov"
    tr_idxs_path = idxs_path + f"/train-{cfg.STEP}.npy"
    te_idxs_path = idxs_path + f"/test-{cfg.STEP}.npy"

    if cfg.dataset == 'voc':
        CLASSES = VOC_CLASSES
        trainset = VOCSegmentationIncremental(
            root=cfg.DATA.ROOT, train=True, transform=tr_transform,
            labels=list(labels), labels_old=list(labels_old),
            idxs_path=tr_idxs_path, overlap=cfg.OVERLAP,
        )
        testset = VOCSegmentationIncremental(
            root=cfg.DATA.ROOT, train=False, transform=val_transform,
            labels=list(labels_cum),
            idxs_path=te_idxs_path,
        )
    elif cfg.dataset == 'ade':
        CLASSES = ADE_CLASSES
        trainset = AdeSegmentationIncremental(
            root=cfg.DATA.ROOT, train=True, transform=tr_transform,
            labels=list(labels), labels_old=list(labels_old),
            idxs_path=tr_idxs_path, overlap=cfg.OVERLAP,
        )
        testset = AdeSegmentationIncremental(
            root=cfg.DATA.ROOT, train=False, transform=val_transform,
            labels=list(labels_cum),
            idxs_path=te_idxs_path,
        )
    else:
        TypeError("Invalid dataset {cfg.dataset}")

    tr_sampler = torch.utils.data.sampler.BatchSampler(
        TrainingSampler(len(trainset)),
        cfg.DATA.BATCH_SIZE,
        drop_last=True,
    )
    te_sampler = torch.utils.data.sampler.BatchSampler(
        InferenceSampler(len(testset)),
        1,
        drop_last=False,
    )

    train_loader = DataLoader(
        trainset, num_workers=4, batch_sampler=tr_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    test_loader = DataLoader(
        testset, num_workers=4, batch_sampler=te_sampler,
        collate_fn=trivial_batch_collator,
    )


    return trainset, testset, train_loader, test_loader, CLASSES
