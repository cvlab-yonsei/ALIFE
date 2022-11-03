import os
import time
import torch
import argparse

### from Detectron2 ###
from configs.defaults import _C

### from MiB/PLOP ###
import utils.tasks as tasks
from models.cayley_rot import Cayley_Rot


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}")

    cfg = _C.clone()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.dataset = args.config_file.split("/")[1]
    cfg.mem_size = args.mem_size
    cfg.mem_name = os.path.join("./checkpoints", cfg.MODEL.WEIGHTS.replace(".pt", f"_M{args.mem_size}.pt"))
    cfg.matrices = f"ROT_{cfg.SEED}_"
    if cfg.OVERLAP:
        cfg.matrices += "ov_"
    else:
        cfg.matrices += "dis_"
    cfg.matrices += f"{cfg.TASK}_{cfg.STEP}_last.pt"

    num_classes = tasks.get_per_task_classes(cfg.dataset, cfg.TASK, cfg.STEP)
    num_cls = sum(num_classes[:cfg.STEP])
    cfg.save_name = os.path.join("./checkpoints", cfg.matrices.replace(".pt", f"_C{num_cls}M{cfg.mem_size}.pt"))
    cfg.freeze()


    print(f"Loading Memory from {cfg.mem_name} ...")
    memory = torch.load(cfg.mem_name, map_location='cpu').to(device) 
    print(f"Memory Size: {memory.shape} (num_cls, num_mem, num_dim)\n")

    print(f"Loading Roation Matrices from {cfg.matrices} ...\n")
    model = Cayley_Rot(num_cls).to(device)
    chkpt = torch.load(f"./checkpoints/{cfg.matrices}", map_location='cpu')
    model.load_state_dict(chkpt)

    new = []
    with torch.no_grad():
        for ind in range(num_cls):
            new += [torch.matmul(model.get_matrix(ind), memory[ind].t()).t()] # (num_mem, num_dim)
        new = torch.stack(new, dim=0).detach().cpu() # (num_cls, num_mem, num_dim)

    print(f"Saving New Memory @ {cfg.save_name} ...\n")
    torch.save(new, cfg.save_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--mem-size", type=int, help="size of memory")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU Index (0 or 1)")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()
    args = get_args()
    main(args)
    print('TOTAL TIME (sec): ', time.time() - start_time)
