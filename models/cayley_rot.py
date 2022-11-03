import numpy as np
import torch
import torch.nn as nn

class Cayley_Rot(nn.Module):
    def __init__(self, num_cls, num_dim=256):
        super().__init__()
        self.num_cls = num_cls
        ngf = int((num_dim-1) * num_dim / 2)
        self.row_ind, self.col_ind = torch.triu_indices(num_dim, num_dim, offset=1)
        self.register_buffer("I_mat", torch.eye(num_dim, dtype=torch.float))
        self.params = nn.ModuleList([nn.Conv2d(ngf,1,1,bias=False) for _ in range(self.num_cls)])

    def get_matrix(self, ind):
        base_mat = torch.zeros_like(self.I_mat) 
        base_mat[self.row_ind, self.col_ind] = self.params[ind].weight.squeeze()
        skewsym_mat = .5 * (base_mat - base_mat.t()) 
        return torch.matmul(self.I_mat - skewsym_mat, torch.inverse(self.I_mat + skewsym_mat))   

    def forward(self, input):
        '''
        input  : (num_dim, num_cls) float32
        output : (num_dim, num_cls) float32
        '''
        out = []
        for ind in range(self.num_cls):
            out += [torch.matmul(self.get_matrix(ind), input[:,ind])]
        return torch.stack(out, dim=1) 
