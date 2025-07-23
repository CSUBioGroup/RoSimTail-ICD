from torch import nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from collections import OrderedDict
np.random.seed(20230620)
def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(random_seed=20230620)


attention_list=[]
class SelfAttention_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, name='selfAttn'):
        super(SelfAttention_PostLN, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(dropout)
        self.name = name
        #===================================================================
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)

    def forward(self, qx, kx, vx, maskPAD=None):
        Bq,qL,C = qx.shape
        B,kvL,C = kx.shape

        queries = self.WQ(qx).reshape(Bq,qL,self.multiNum,self.dk).transpose(1,2)
        keys    = self.WK(kx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2)
        values  = self.WV(vx).reshape(B,kvL,self.multiNum,self.dk).transpose(1,2)
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk)

        alpha = F.softmax(scores, dim=3)
        
        z = self.dropout(alpha) @ values
        z = z.transpose(1,2).reshape(B,qL,-1)
        z = self.WO(z)
        return z


class FFN_PostLN(nn.Module):
    def __init__(self, feaSize, dropout=0.1, actFunc=nn.GELU, name='FFN'):
        super(FFN_PostLN, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])

    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z))
        return self.layerNorm2(z+self.dropout(z))
    
    
#
class Transformer_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PostLN, self).__init__()
        self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum)
        self.ffn = FFN_PostLN(feaSize, dropout)
    
    def forward(self, input):
        qx,kx,vx, maskPAD = input 
        z = self.selfAttn(qx, kx, vx, maskPAD)
        return (self.ffn(qx, z), kx, vx, maskPAD)

class TransformerLayers_PostLN(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, dropout=0.1, name='textTransformer'):
        super(TransformerLayers_PostLN, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_PostLN(feaSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
        
    def forward(self, qx, kx, vx, maskPAD=None):
        
        return self.transformerLayers((qx,kx,vx, maskPAD))