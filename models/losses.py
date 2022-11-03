import torch
import torch.nn as nn
import torch.nn.functional as F 


def CrossEntropyLoss(ignore_index=255, reduction='mean'):
    return nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index=ignore_index
        self.size_average=size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
        
class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss


class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):
        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss # B, H, W
        return outputs


def get_losses(cfg, use_mem):
    if cfg.LOSS.CE.TYPE in ["ce", ""]:
        criterion_CE = CrossEntropyLoss(ignore_index=255, reduction='none')
    elif cfg.LOSS.CE.TYPE == "uce":
        import utils.tasks as tasks
        _, labels_old = tasks.get_task_labels(cfg.dataset, cfg.TASK, cfg.STEP)
        criterion_CE = UnbiasedCrossEntropy(old_cl=len(labels_old), ignore_index=255)
    elif cfg.LOSS.CE.TYPE == "fl":
        criterion_CE = FocalLoss()
    else:
        TypeError(f"Unknown cfg.LOSS.CE.TYPE {cfg.LOSS.CE.TYPE}")

    if cfg.LOSS.KD.TYPE in ["kd", ""]:
        criterion_KD = KnowledgeDistillationLoss(reduction='none')
    elif cfg.LOSS.KD.TYPE == "ukd":
        criterion_KD = UnbiasedKnowledgeDistillationLoss(reduction='none')
    else:
        TypeError(f"Unknown cfg.LOSS.KD.TYPE {cfg.LOSS.KD.TYPE}")

    if cfg.LOSS.MEMORY.TYPE in ["ce", ""]:
        criterion_MEM = CrossEntropyLoss()
    elif cfg.LOSS.MEMORY.TYPE == "fl":
        criterion_MEM = FocalLoss()
    else:
        TypeError(f"Unknown cfg.LOSS.MEMORY.TYPE {cfg.LOSS.MEMORY.TYPE}")

    if use_mem:
        return criterion_CE, criterion_MEM
    return criterion_CE, criterion_KD
