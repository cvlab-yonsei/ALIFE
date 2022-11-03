import os
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

### from Detectron2 ###
from configs.defaults import _C

### from MiB/PLOP ###
import utils.tasks as tasks
from datasets import *
from models.deeplabv3 import DeepLabV3


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")

    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.dataset = args.config_file.split("/")[1]
    # NOTE that ALIFE-S1 and -S3 have the same feature extractor in the same step.
    cfg.weights = cfg.MODEL.WEIGHTS 
    cfg.mem_size = args.mem_size
    cfg.save_name = os.path.join("./checkpoints", cfg.weights.replace(".pt", f"_M{args.mem_size}.pt"))
    cfg.STEP = cfg.STEP - 1 
    cfg.freeze()


    ## MODEL
    print(f"Loading a feature extractor from {cfg.weights} ...")
    num_classes = tasks.get_per_task_classes(cfg.dataset, cfg.TASK, cfg.STEP)
    model = DeepLabV3(num_classes, freeze_type="all").to(device)
    missing, unexpected = model.load_state_dict(torch.load(f"./checkpoints/{cfg.weights}", map_location="cpu"), strict=False)
    print(f"Missing    : {missing}")
    print(f"Unexpected : {unexpected}\n")


    ## DATA
    trainset, _, _, _, CLASSES = get_datasets(cfg)
    labels, labels_old = tasks.get_task_labels(cfg.dataset, cfg.TASK, cfg.STEP)
    if cfg.dataset == "voc":
        mask_path = os.path.join(cfg.DATA.ROOT, 'SegmentationClassAug/{}.png')
        file_list = os.path.join(cfg.DATA.ROOT, 'ImageSets/Segmentation/train_aug.txt')
        file_list = [fn.split('\n')[0] for fn in open(file_list, 'r')]
    if cfg.dataset == "ade":
        mask_path = os.path.join(cfg.DATA.ROOT, 'annotations/training/{}.png')
        file_list = sorted(os.listdir(os.path.join(cfg.DATA.ROOT, 'images/training/')))
        file_list = [fn.split('.jpg')[0] for fn in file_list]

    print("Target transform (Train): {}".format(trainset.target_transform.tolist()))

    print("The numbers of valid images for each category")
    per_cat_indices = {cat: [] for cat in labels}
    for it, ind in enumerate(trainset.dataset.indices):
        mask = np.array(Image.open(mask_path.format(file_list[ind])))
        uni_classes = np.unique(mask).tolist()
        for uni_cat in uni_classes:
            if uni_cat in labels:
                ratio = (mask==uni_cat).sum() / (mask.shape[0]*mask.shape[1])
                if ratio > 0.005: # To avoid too small objects
                    per_cat_indices[uni_cat] += [it] 

    for cat in labels:
        cls_info = f"{cat:2d}-{CLASSES[cat]:.11}"
        print(f"\t{cls_info:<14}: {len(per_cat_indices[cat]):5d}")
    print("\n")


   ## Extract
    memory_bank = {cat: [] for cat in labels}
    with torch.no_grad():
        model.eval()
        for cat in labels:
            pbar = tqdm(total=cfg.mem_size)
            while len(memory_bank[cat]) < cfg.mem_size:
                sample_id = np.random.choice(per_cat_indices[cat]) # Sampling with replacement
                img, mask = trainset[sample_id]

                features = model.get_features(img[None].to(device))
                features = F.interpolate(features, mask.shape, mode='bilinear', align_corners=True)
                features = features.squeeze().detach().cpu() # (256, 512, 512)

                region = mask==cat
                avg_feature = features[:,region].mean(dim=1) # (256,)
                ratio = region.sum() / (mask.shape[0]*mask.shape[1])

                if torch.isnan(avg_feature).sum() or ratio < 0.005: # Due to DataAug, there could be nan or too small objects
                    continue
                else:
                    memory_bank[cat].append(avg_feature)
                    pbar.update(1)
            pbar.close()
            print(f"\tCategory-{cat} is Done")
    print("\n")
    memory_bank = torch.stack([torch.stack(memory_bank[cat], dim=0) for cat in labels], dim=0)


    ## Save
    if cfg.STEP > 0: # load previous memory
        num_cls = sum(num_classes[:cfg.STEP])

        mem_name = f"ROT_{cfg.SEED}_"
        if cfg.OVERLAP:
            mem_name += "ov_"
        else:
            mem_name += "dis_"
        mem_name += f"{cfg.TASK}_{cfg.STEP}_last_C{num_cls}M{cfg.mem_size}.pt"
        mem_name = os.path.join("./checkpoints", mem_name)
        print(f"Loading previous memory from {mem_name} ...")
        old_mem = torch.load(mem_name, map_location="cpu")
        print(f"Size of previous memory : {old_mem.shape} (num_cls, num_mem, num_dim)\n")
        memory_bank = torch.cat([old_mem, memory_bank], dim=0)
    torch.save(memory_bank, cfg.save_name)
    print(f"Saving New Memory @ {cfg.save_name} ...")
    print(f"Size of new memory : {memory_bank.shape} (num_cls, num_mem, num_dim)\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--mem-size", type=int, help="size of memory")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU Index (0 or 1)")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()
    main(args)
    print(f"TOTAL TIME (sec): {time.time() - start_time}\n")
