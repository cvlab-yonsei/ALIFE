import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

### from Detectron2 ###
import utils.comm as comm
from configs.defaults import _C
from utils.engine import launch
from utils.checkpointer import Checkpointer
from utils.sem_seg_evaluation import SemSegEvaluator
from utils.distributed_sampler import seed_all_rng 

### from MiB/PLOP ###
import utils.tasks as tasks
from datasets import *
from models.deeplabv3 import DeepLabV3
from models.cayley_rot import Cayley_Rot
from models.losses import CrossEntropyLoss

logger = logging.getLogger("train_matrices")

def do_train(cfg, FEprev, FEcurr, model, memory):
    logger.info(model)

    trainset, _, train_loader, _, _ = get_datasets(cfg)

    iters_per_epoch = len(trainset) // (cfg.DATA.BATCH_SIZE * cfg.num_gpus)
    max_iter = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    lr_lambda = lambda it: (1-it/max_iter)**cfg.SOLVER.GAMMA
    optimizer = torch.optim.Adam(get_params(model), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = CrossEntropyLoss()

    logger.info("<Previous feature extractor>")
    checkpointer_FEprev = Checkpointer(FEprev, "train_matrices")
    checkpointer_FEprev.load(cfg.MODEL.WEIGHTS)
    logger.info("<Current feature extractor>")
    checkpointer_FEcurr = Checkpointer(FEcurr, "train_matrices")
    checkpointer_FEcurr.load(cfg.CURR_WEIGHTS)
    FEprev.eval()
    FEcurr.eval()

    _, labels_old = tasks.get_task_labels(cfg.dataset, cfg.TASK, cfg.STEP)
    num_cls = len(labels_old)
    num_mem = cfg.mem_size
    assert num_mem == memory.shape[1]
    normed_mem = F.normalize(memory.view(num_cls*num_mem,-1), dim=1)
    mem_gt = torch.tensor(labels_old, dtype=torch.long).to(torch.device("cuda"))
    mem_gt = mem_gt.unsqueeze(-1).unsqueeze(-1) # (num_cls, 1, 1)

    logger.info(f"Trainset Size: {len(trainset)}")
    logger.info("Target transform (Train) : {}".format(trainset.target_transform.tolist()))
    logger.info(f"START {cfg.SAVE_NAME} -->")

    model.train()
    TEMP = cfg.TEMP
    LAMBDA = cfg.LAMBDA
    ep=1
    interval_eval = 1
    interval_verbose = iters_per_epoch // 10
    storages = {"Total": 0, "CE": 0, "CS": 0}
    for it, batch in zip(range(1, max_iter+1), train_loader):
        img = torch.stack([x[0] for x in batch], dim=0).to(torch.device("cuda"))  
        gt  = torch.stack([x[1] for x in batch], dim=0)

        loss_dict = {}
        with torch.no_grad():
            fea_prev = FEprev.get_features(img)
            fea_curr = FEcurr.get_features(img)
            bs,dims,fH,fW = fea_curr.shape

            fea_prev = fea_prev.view(bs,dims,-1).permute(0,2,1).contiguous().view(-1,dims)
            fea_curr = fea_curr.view(bs,dims,-1).permute(0,2,1).contiguous().view(-1,dims)

            normed_prev = F.normalize(fea_prev, dim=1) # (bs*fH*fW, dims)
            
            weights = torch.matmul(normed_mem, normed_prev.t()) # (num_cls*num_mem, bs*fH*fW)
            weights = F.relu(weights)
            weights = weights.view(num_cls,num_mem,-1)
            weights = torch.sum(weights, dim=1) # (num_cls, bs*fH*fW)
            weights = torch.softmax(TEMP * weights, dim=1)

            rep_prev = torch.matmul(weights, fea_prev) # (num_cls, dims)
            rep_curr = torch.matmul(weights, fea_curr)

            ## centering
            off_prev = rep_prev.mean(dim=0, keepdim=True) # (1, dims)
            off_curr = rep_curr.mean(dim=0, keepdim=True) 
            rep_prev = rep_prev - off_prev
            rep_curr = rep_curr - off_curr

        rep_hat = model(rep_prev.t()).t() # (num_cls, dims)

        logits = FEcurr.get_prediction((rep_hat+off_curr).unsqueeze(-1).unsqueeze(-1))
        similarity = F.cosine_similarity(rep_hat, rep_curr, dim=1)

        loss_dict["loss_ce"] = (1-LAMBDA) *criterion(logits, mem_gt)
        loss_dict["loss_cs"] = LAMBDA * torch.mean(1 - similarity)
        losses = sum(loss_dict.values())
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()
        storages["Total"] += losses_reduced 
        storages["CE"] += loss_dict_reduced["loss_ce"]
        storages["CS"] += loss_dict_reduced["loss_cs"]

        if it % interval_verbose == 0:
            verbose = f"{it:5d}/{max_iter+1:5d}  CE: {loss_dict_reduced['loss_ce']:.4f}  CS: {loss_dict_reduced['loss_cs']:.4f}"
            logger.info(verbose)

        if it % iters_per_epoch == 0:
            for k in storages.keys(): storages[k] /= it
            logger.info("epoch: {:3d}  Total: {:.4f}  CE: {:.4f}  CS: {:.4f}  lr: {}\n".format(ep, storages["Total"], storages["CE"], storages["CS"], optimizer.param_groups[0]["lr"])) 
            for k in storages.keys(): storages[k] = 0

            ep += 1

    if comm.is_main_process():
        torch.save(model.module.state_dict(), f"./checkpoints/{cfg.SAVE_NAME}_last.pt")
    logger.info(f"END {cfg.SAVE_NAME} -->")


def main(args):
    start_time = time.time()

    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    cfg.num_gpus = args.num_gpus
    cfg.dataset  = args.config_file.split("/")[1]
    cfg.mem_size = args.mem_size
    cfg.CURR_WEIGHTS = f"{cfg.TAG}_{cfg.SEED}_"
    if cfg.OVERLAP: 
        cfg.CURR_WEIGHTS += "ov_"
    else:
        cfg.CURR_WEIGHTS += "dis_"
    cfg.CURR_WEIGHTS += f"{cfg.TASK}_{cfg.STEP}_last.pt"
    cfg.TEMP = args.temp
    cfg.LAMBDA = args.lamb

    save_name = f"ROT_{cfg.SEED}" 
    if cfg.OVERLAP:
        save_name += "_ov"
    else:
        save_name += "_dis"
    cfg.SAVE_NAME = save_name + f"_{cfg.TASK}_{cfg.STEP}" 
    cfg.freeze()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if comm.is_main_process():
        formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(f"./logs/{cfg.SAVE_NAME}.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info(" ".join(["\n{}: {}".format(k, v) for k,v in cfg.items()]))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + comm.get_rank())
    num_classes = tasks.get_per_task_classes(cfg.dataset, cfg.TASK, cfg.STEP)
    FEprev = DeepLabV3(num_classes[:-1], cfg.MODEL.SYNC_BN, freeze_type="all").to(torch.device("cuda"))
    FEcurr = DeepLabV3(num_classes, cfg.MODEL.SYNC_BN, freeze_type="all").to(torch.device("cuda"))
    FEprev.eval()
    FEcurr.eval()

    mem_name = os.path.join("./checkpoints", cfg.MODEL.WEIGHTS.replace(".pt", f"_M{args.mem_size}.pt"))
    logger.info(f"Loading Memory from {mem_name} ...")
    has_file = os.path.isfile(mem_name)
    all_has_file = comm.all_gather(has_file)
    if not all_has_file[0]:
        raise RuntimeError(f"There is no {mem_name}")
    memory = torch.load(mem_name, map_location='cpu').to(torch.device("cuda")) 
    logger.info(f"Memory Size: {memory.shape} (num_cls, num_mem, num_dim)")

    model = Cayley_Rot(sum(num_classes[:-1])).to(torch.device("cuda"))
    if args.num_gpus > 1:
        model = DDP(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

    do_train(cfg, FEprev, FEcurr, model, memory)
    
    tt = time.time() - start_time
    hours = int(tt // 3600)
    mins  = int((tt % 3600) // 60)
    logger.info(f"ELAPSED TIME: {hours:02d}(h) {mins:02d}(m)")


def get_params(model):
    params = []
    for name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if value.requires_grad:
                logger.info(f"Learning {name}-{module_param_name}")
                params.append({"params": [value]})
    return params


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--mem-size", type=int, help="size of memory")
    parser.add_argument("--temp", type=float, default=10., help="temperature controlling the sharpness")
    parser.add_argument("--lamb", type=float, default=0.99, help="balance parameter")
    parser.add_argument("--num-gpus", type=int, default=2, help="number of gpus *per machine*")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    launch(main, args.num_gpus, args=(args,)) 
