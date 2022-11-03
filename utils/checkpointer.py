import os
import torch
import logging
import utils.comm as comm

class Checkpointer:
    def __init__(self, model, name, save_path="./checkpoints"):
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            model = model.module
        self.model = model
        self.logger = logging.getLogger(name)
        self.save_path = save_path
        self.save_to_disk = True if comm.is_main_process() else False
        self.best_performance = 0

    def load(self, path, is_backbone=False):
        file_path = f"{self.save_path}/{path}"
        self.logger.info(f"Loading from {file_path} ...")
        has_file = os.path.isfile(file_path)
        all_has_file = comm.all_gather(has_file)
        if not all_has_file[0]:
            raise RuntimeError(f"There is no {file_path}")
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        if is_backbone:
            missing, unexpected = self.model.backbone.load_state_dict(checkpoint, strict=False)
        else:
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
        self.logger.info(f"Missing    : {missing}")
        self.logger.info(f"Unexpected : {unexpected}")

    def save(self, name, performance=None):
        if not self.save_to_disk:
            return
        if performance is not None and performance["mIoU"] > self.best_performance:
            self.best_performance = performance["mIoU"]
            data = self.model.state_dict()
            torch.save(data, f"{self.save_path}/{name}.pt")
            self.logger.info("--- SAVED ---")
 
