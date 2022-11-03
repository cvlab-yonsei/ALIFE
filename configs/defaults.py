from yacs.config import CfgNode as CN

_C = CN()

_C.TAG = ""
_C.SAVE_NAME = ""

_C.TASK = ""
_C.STEP = 0 
_C.OVERLAP = True

_C.SEED = -1 

_C.DATA = CN()
_C.DATA.ROOT = ""
_C.DATA.BATCH_SIZE = 0
_C.DATA.CROP_SIZE = 0

_C.MODEL = CN()
_C.MODEL.WEIGHTS = ""
_C.MODEL.SYNC_BN = False
_C.MODEL.FREEZE_TYPE = ""
_C.MODEL.MIB_CLS_INIT = False

_C.LOSS = CN()
_C.LOSS.CE = CN()
_C.LOSS.CE.TYPE = ""
_C.LOSS.MY = CN()
_C.LOSS.MY.WEIGHT = 0.
_C.LOSS.KD = CN()
_C.LOSS.KD.TYPE = ""
_C.LOSS.KD.WEIGHT = 0.
_C.LOSS.MEMORY = CN()
_C.LOSS.MEMORY.TYPE = "" 
_C.LOSS.MEMORY.WEIGHT = 0.

_C.SOLVER = CN()
_C.SOLVER.LR = 0.
_C.SOLVER.MOMENTUM = 0.
_C.SOLVER.NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.
_C.SOLVER.MAX_EPOCH = 0
_C.SOLVER.GAMMA = 0.
