# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class Cross_MultiAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)


    def forward(self, x, comment_tree, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param comment_tree: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h = x.shape
        batch_size = b
        # x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(comment_tree)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(x)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        # print(att_weights.shape)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]

        # print(out.shape)

        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        # out = self.proj_out(out)   # [batch_size, c, h, w]

        return out, att_weights


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1 * config.hidden_size, 2 * config.hidden_size)
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- mlp ----------------------
        x = torch.tanh(x)
        x = self.dropout(x)        
        x = self.out_proj(x)
        
        return x
     
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier = RobertaClassificationHead(config)
        self.cross_att =Cross_MultiAttention(emb_dim=768, num_heads=8, att_dropout=0.0, aropout=0.0)
        # Define dropout layer, dropout_probability is taken from args.
        # self.dropout = nn.Dropout(args.dropout_probability)
        
    def forward(self, input_ids=None, symbolic_input_ids = None, labels=None):
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1),output_hidden_states=True,return_dict=True)
        # logits=outputs[0]
        # cls_first = torch.mean(outputs.hidden_states[1][:,:,:],dim = 1)
        cls_last = outputs.hidden_states[-1][:,:,:]

        # comment_outputs=self.encoder(comment_input_ids,attention_mask=comment_input_ids.ne(1),output_hidden_states=True,return_dict=True)
        # comment_cls_first = torch.mean(comment_outputs.hidden_states[1][:,:,:],dim = 1)
        # comment_cls_last = torch.mean(comment_outputs.hidden_states[-1][:,:,:],dim = 1)

        symbolic_outputs=self.encoder(symbolic_input_ids,attention_mask=symbolic_input_ids.ne(1),output_hidden_states=True,return_dict=True)
        # symbolic_cls_first = torch.mean(symbolic_outputs.hidden_states[1][:,:,:],dim = 1)
        symbolic_cls_last = symbolic_outputs.hidden_states[-1][:,:,:]
        
        output, _ = self.cross_att(cls_last, symbolic_cls_last)
        output = torch.mean(output, dim = 1)

        output = output.squeeze(dim = 1)
        
        logits = self.classifier(output)
        prob=torch.sigmoid(logits)
        # prob = prob.reshape(prob.shape[0],-1)

        
        # cross entropy loss for classifier

        if labels is not None:
            labels=labels.float()
            loss = 1.2 * torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss = -loss.mean()

            return loss,prob
        else:
            return prob

      
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
 

