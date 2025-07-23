import torch
from torch import nn
from src.transformer import *

class AttentionLayer(nn.Module):

    def __init__(self, args, vocab):

        super(AttentionLayer, self).__init__()

        # ===========================================
        self.attention_mode = args.attention_mode
        self.size = args.hidden_size*2  #768
        self.d_a = args.d_a
        self.n_labels = vocab.label_num

        # if self.attention_mode == "caml":
        #     self.d_a = self.size
        self.tanh = nn.Tanh()
        self.first_linears = nn.Linear(self.size, self.d_a, bias=False)
        self.second_linears = nn.Linear(self.d_a, self.n_labels, bias=False)
        self.third_linears =nn.Linear(self.size,self.n_labels, bias=True)

        self.transformer_layer = TransformerLayers_PostLN(layersNum=1, feaSize=self.size, dk=args.dk,multiNum=args.multiNum,dropout=args.trans_drop)
        self._init_weights(mean=0.0, std=0.03)

    #初始化线性层
    def _init_weights(self, mean=0.0, std=0.03):
        torch.nn.init.normal_(self.first_linears.weight, mean, std)
        
        if self.first_linears.bias is not None:
            self.first_linears.bias.data.fill_(0)
        torch.nn.init.normal_(self.second_linears.weight, mean, std)
        
        if self.second_linears.bias is not None:
            self.second_linears.bias.data.fill_(0)
        torch.nn.init.normal_(self.third_linears.weight, mean, std)

    def forward(self, x , label_batch=None):

        #label_batch.shape torch.Size([1, 50, 10])
        # x.shape torch.Size([8, 10, 10])
        #代码中跑的就是text_label
        if self.attention_mode == "text_label":
            att = self.transformer_layer(qx=label_batch, kx=x, vx=x)#标签描述信息与文本特征进行注意力计算
            att = att[0]
            #Attention weights (att) shape: torch.Size([1, 50, 768])
            #Weighted output shape: torch.Size([1, 50])
            #weighted_output1 = self.third_linears.weight.mul(att)#[1, 50, 768]
            #weighted_output2 = weighted_output1.sum(dim=2).add(self.third_linears.bias)#[1, 50]
            weighted_output = self.third_linears.weight.mul(att).sum(dim=2).add(self.third_linears.bias)#[1,50]
            logits = weighted_output.reshape(weighted_output.shape[0], -1)
            
            return att,logits


        elif self.attention_mode == "laat":
            weights = self.tanh(self.first_linears(x))
            att_weights = self.second_linears(weights)
            att_weights = F.softmax(att_weights, 1).transpose(1, 2)
            if len(att_weights.size()) != len(x.size()):
                att_weights = att_weights.squeeze()
            # 8*8929*2506,8*2506*1024，softmax是lenth维度，就是每一个标签的2506个不同分数加起来为1，表示每个词对这个标签的相关度，
            # 相乘后得到每一个标签的1024个特征，这些特征都是在这一特征维度词的分数加权和
            # att = att_weights @ x
            weighted_output = att_weights @ x
            weighted_output = self.third_linears.weight.mul(weighted_output).sum(dim=2).add(self.third_linears.bias)
            logits = weighted_output.reshape(weighted_output.shape[0], -1)

        return logits