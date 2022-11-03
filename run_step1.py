import sys
import time
import logging
import argparse
import numpy as np
import torch
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
from datasets import *
import utils.tasks as tasks
from models.deeplabv3 import DeepLabV3
from models.losses import get_losses

logger = logging.getLogger("Step 1")

def do_test(cfg, model, logger, checkpointer, testset, test_loader, CLASSES):
    evaluator = SemSegEvaluator(len(CLASSES), distributed=True) 
    evaluator.reset()
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            img = torch.stack([x[0] for x in batch], dim=0).to(torch.device("cuda"))  
            gt  = torch.stack([x[1] for x in batch], dim=0).numpy()
            logits = model(img)
            pred = logits.argmax(dim=1).to(torch.device("cpu")).numpy() 
            evaluator.process(pred, gt)
    results = evaluator.evaluate()

    if comm.is_main_process():
        logger.info("# of Test Samples: {}".format(results["Total samples"]))
        logger.info("{:>21}\t{:>}".format("<MA>", "<IoU>"))
        prev_ma , curr_ma  = [], []
        prev_iou, curr_iou = [], []
        base_classes = tasks.get_tasks(cfg.dataset, cfg.TASK, 0)
        num_base_cls = len(base_classes)
        target_cat_list = testset.target_transform.tolist()
        for ind, (ma, iou) in enumerate(zip(results["per Acc"], results["per IoU"])):
            if ind in target_cat_list:
                cat_ind = target_cat_list.index(ind) 
                cls_info = f"{ind:2d}-{CLASSES[cat_ind]:.11}"
                logger.info(f"{cls_info:<14}: {ma:05.2f}\t{iou:05.2f}")
                if cat_ind in base_classes:
                    prev_ma  += [ma]
                    prev_iou += [iou]
                else:
                    curr_ma  += [ma]
                    curr_iou += [iou]

        prev_ma = np.nanmean(prev_ma) if len(prev_ma) else np.nan
        curr_ma = np.nanmean(curr_ma) if len(curr_ma) else np.nan
        prev_iou = np.nanmean(results["per IoU"][:num_base_cls])
        curr_iou = np.nanmean(results["per IoU"][num_base_cls:])
        hIoU = 2*prev_iou*curr_iou/(prev_iou+curr_iou)

        #PA   = results["pACC"]
        #mMA  = results["mACC"]
        mIoU = results["mIoU"]

        #logger.info(f"PA: {PA:.2f}  mMA: {mMA:.2f}")
        logger.info(f" prev-MA: {prev_ma:.2f}   curr-MA: {curr_ma:.2f}")
        logger.info(f"prev-IoU: {prev_iou:.2f}  curr-IoU: {curr_iou:.2f}  mIoU: {mIoU:.2f}  hIoU: {hIoU:.2f}")
 
    if results is None: results = {}
    return results


def do_train(cfg, model, model_old=None):
    logger.info(model)

    trainset, testset, train_loader, test_loader, CLASSES = get_datasets(cfg)

    if cfg.eval_only:
        logger.info("<Testing model>")
        checkpointer = Checkpointer(model, "Step 1")
        checkpointer.load(cfg.weights)
        scores = do_test(cfg, model, logger, checkpointer, testset, test_loader, CLASSES)
        return scores

    iters_per_epoch = len(trainset) // (cfg.DATA.BATCH_SIZE * cfg.num_gpus)
    max_iter = iters_per_epoch * cfg.SOLVER.MAX_EPOCH
    lr_lambda = lambda it: (1-it/max_iter)**cfg.SOLVER.GAMMA
    optimizer = optim.SGD(
        get_params(model), 
        lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, 
        nesterov=cfg.SOLVER.NESTEROV, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) 
    criterion_CE, criterion_KD = get_losses(cfg, use_mem=False)

    if model_old is not None:
        logger.info("<Previous model>")
        checkpointer_old = Checkpointer(model_old, "Step 1")
        checkpointer_old.load(cfg.MODEL.WEIGHTS)
        model_old.eval()
    logger.info("<Current model>")
    checkpointer = Checkpointer(model, "Step 1")
    checkpointer.load(cfg.MODEL.WEIGHTS, bool(cfg.STEP==0))

    if cfg.MODEL.MIB_CLS_INIT:
        if isinstance(model, DDP):
            model.module._init_like_MiB() 
        else:
            model._init_like_MiB() 

    logger.info("Trainset Size: {}".format(len(trainset)))
    logger.info("Target transform (Train) : {}".format(trainset.target_transform.tolist()))
    logger.info("Target transform  (Test) : {}".format(testset.target_transform.tolist()))
    logger.info(f"START {cfg.SAVE_NAME} -->")

    model.train()
    ep = 1
    interval_eval = cfg.SOLVER.MAX_EPOCH
    interval_verbose = iters_per_epoch // 10
    storages = {"Total": 0, "CE": 0, "ALI": 0, "KD": 0} 
    for it, batch in zip(range(1, max_iter+1), train_loader):
        img = torch.stack([x[0] for x in batch], dim=0).to(torch.device("cuda"))  
        gt  = torch.stack([x[1] for x in batch], dim=0).to(torch.device("cuda")) 

        logits = model(img)
        loss_dict = {}
        if cfg.STEP > 0:
            with torch.no_grad():
                logits_old = model_old(img) 
                prob_old = torch.softmax(logits_old, dim=1) # (bs, C_old, H, W) 
                pred_old = logits_old.argmax(dim=1)
            bg_region = gt==0
            hihohu = torch.logsumexp(logits, dim=1) - torch.sum(prob_old * logits[:,:logits_old.shape[1]], dim=1) # (bs, H, W) 
            loss_dict["loss_ali"] = cfg.LOSS.MY.WEIGHT * hihohu[bg_region].mean() 
            loss_dict["loss_kd"]  = cfg.LOSS.KD.WEIGHT * criterion_KD(logits, logits_old)[~bg_region].mean()

            gt[gt==0] = 255 # NOTE: This is to use CE (delete it when using UCE)
        loss_dict["loss_ce"] = criterion_CE(logits, gt).mean() 

        losses = sum(loss_dict.values())
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step() 
        storages["Total"] += losses_reduced 
        storages["CE"] += loss_dict_reduced["loss_ce"]
        if model_old is not None:
            storages["ALI"] += loss_dict_reduced["loss_ali"]
            storages["KD"] += loss_dict_reduced["loss_kd"]

        if it % interval_verbose == 0:
            verbose = f"{it:5d}/{max_iter+1:5d}  CE: {loss_dict_reduced['loss_ce']:.4f}  "
            if cfg.STEP > 0:
                verbose += f"ALI: {loss_dict_reduced['loss_ali']:.4f}  KD: {loss_dict_reduced['loss_kd']:.4f}"
            logger.info(verbose)

        if it % iters_per_epoch == 0:
            for k in storages.keys(): storages[k] /= it
            logger.info(
                "epoch: {:3d}  Total: {:.4f}  CE: {:.4f}  ALI: {:.4f}  KD: {:.4f}  lr: {}".format(
                    ep, storages["Total"], storages["CE"], storages["ALI"], storages["KD"], optimizer.param_groups[0]["lr"]
                )
            )
            for k in storages.keys(): storages[k] = 0

            if ep % interval_eval == 0:
                scores = do_test(cfg, model, logger, checkpointer, testset, test_loader, CLASSES)
                model.train()
                comm.synchronize()
                logger.info("\n")

            ep += 1

    checkpointer.save(cfg.SAVE_NAME+"_last", scores) 
    logger.info(f"END {cfg.SAVE_NAME} -->")


def main(args):
    start_time = time.time()

    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts) 
    cfg.num_gpus = args.num_gpus
    cfg.dataset  = args.config_file.split("/")[1]
    cfg.eval_only = args.eval_only
    cfg.weights = args.weights
    save_name = f"{cfg.TAG}_{cfg.SEED}_" 
    if cfg.OVERLAP:
        save_name += "ov_"
    else:
        save_name += "dis_"
    save_name += f"{cfg.TASK}_{cfg.STEP}"
    cfg.SAVE_NAME = save_name
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
    model = DeepLabV3(num_classes, cfg.MODEL.SYNC_BN, freeze_type=cfg.MODEL.FREEZE_TYPE).to(torch.device("cuda"))
    model_old = None
    if cfg.STEP > 0:
        model_old = DeepLabV3(num_classes[:-1], cfg.MODEL.SYNC_BN, freeze_type="all").to(torch.device("cuda"))
        model_old.eval()

    if args.num_gpus > 1:
        model = DDP(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=False) 

    do_train(cfg, model, model_old)

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
    parser.add_argument("--eval-only", action="store_true", default=False, help="perform evaluation only")
    parser.add_argument("--weights", type=str, default="", help="pre-trained weights for evaluation only")
    parser.add_argument("--num-gpus", type=int, default=2, help="number of gpus")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    launch(main, args.num_gpus, args=(args,)) 
