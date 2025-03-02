import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqAttention(nn.Module):
    def __init__(self, configs):
        super(SeqAttention, self).__init__()
        self.d_model = configs.seq_len // 2
        # d_model=seq_len
        self.w_qs = nn.Linear(self.d_model, self.d_model)
        self.w_ks = nn.Linear(self.d_model, self.d_model)
        self.batch_size = None
        self.seq_number = configs.seq_number
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_seq, key_seqs):
        # query_seq: [batch,column,seqlen//2],key_seqs: [batch,seq_number,seqlen//2]
        # query_seq = query_seq[:, 1:]
        # key_seqs = key_seqs[:, 1:]
        self.batch_size=query_seq.shape[0]
        device = query_seq.device
        query_seq = query_seq.abs().float().to(device)
        key_seqs = key_seqs.abs().float().to(device)

        query_seq_mean = query_seq.mean()
        query_seq_std = query_seq.std()
        key_seqs_mean = key_seqs.mean()
        key_seqs_std = key_seqs.std()

        query_seq = (query_seq - query_seq_mean) / query_seq_std
        key_seqs = (key_seqs - key_seqs_mean) / key_seqs_std

        q = self.w_qs(query_seq)  # [batch,column,seqlen//2]

        # Prepare K from the key sequences

        k = self.w_ks(key_seqs)  # [batch,seq_number,seqlen//2]
        attn_scores = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model))
        if self.seq_number == query_seq.shape[1]:
            mask = torch.eye(self.seq_number, self.seq_number, dtype=torch.bool,device=query_seq.device)
            mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1)
            attn_scores = attn_scores.masked_fill(mask, -1e7)
        # [batch,column,seq_number]
        # print(attn_scores.shape)
        attn_weights = self.softmax(attn_scores)
        v = torch.matmul(attn_weights, key_seqs)
        return attn_weights,v


class SeqModel(nn.Module):
    def __init__(self, configs):
        super(SeqModel, self).__init__()
        self.attention_layer = SeqAttention(configs)
        self.num_layers = 2

    def forward(self, query_seq, key_seqs):
        attn_weights_list = []
        current_input = query_seq

        for _ in range(self.num_layers):
            attn_weights, v = self.attention_layer(current_input, key_seqs)
            current_input = current_input + v
            attn_weights_list.append(attn_weights)

        return attn_weights_list[-1]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = None
        self.random_size = configs.random_size
        self.seq_number = configs.seq_number
        self.channels = configs.enc_in
        self.myattention = SeqModel(configs)
        self.stacked_tensor = None
        self.attn_weights = None
        self.cnn_weights = None
        self.x_liner = nn.Linear(self.seq_len, self.seq_len)
        self.random_liner = nn.Linear(self.random_size, self.random_size)
        self.Linear = nn.Linear(self.pred_len, self.pred_len)
        self.device = None
        self.highway = nn.Linear(self.seq_len, self.pred_len)
        self.stride = configs.stride
        self.action=nn.ReLU()

    def get_stacked_tensor(self, stacked_tensor):
        self.stacked_tensor = stacked_tensor.to(self.device)

    def forward(self, x, random):

        self.device = x.device
        self.batch_size = x.shape[0]
        y1 = self.highway(x)

        self.channels = x.shape[1]
        random = random.to(self.device)

        key_seqs = torch.stack([self.stacked_tensor] * self.batch_size)
        # print("key_seqs.requires_grad after extract_row:", key_seqs.requires_grad)
        query_seq = torch.fft.fft(x, dim=-1)
        # print("query_seq.requires_grad after FFT:", query_seq.requires_grad)
        len1 = query_seq.shape[2] // 2
        query_seq = query_seq[:, :, 1:len1 + 1]
        # query_seq = torch.from_numpy(query_seq)

        self.attn_weights = self.myattention(query_seq, key_seqs)
        # attn_weights[batch,column,seq_number]
        random = random.transpose(-1, -2).float()
        x = self.x_liner(x).float()
        random = self.random_liner(random)

        # random[batch,seq_number,random_number]
        # x[batch,seq_number,self.seq_len]
        input_2d = random[:, :, :-self.pred_len].unfold(-1, self.seq_len, self.stride)
        # input_2d[batch,seq_number,random_number-pred_len+1-seq_len,seq_len]
        # stride:[batch,seq_number,(random_number-pred_len+1-seq_len)/stride,seq_len]
        kernel = x.unsqueeze(1).expand(-1, self.seq_number, -1, -1)
        # kernel[batch,seq-number,column,self.seq_len]
        conv_result = torch.einsum('ijkl,ijml->ijmk', input_2d, kernel).squeeze()
        softmax_result = F.softmax(conv_result, dim=-1)
        # softmax_result :[batch,seq_number,column,random_number-seq_number-pred_number+1]
        tensor_2d = random[:, :, self.seq_len:]
        sliced_tensor = tensor_2d.unfold(dimension=-1, size=self.pred_len, step=self.stride)
        # sliced_tensor_correct:[batch,seq_number, random_number - seq_len - pred_len + 1,pred_len]
        self.cnn_weights = torch.einsum('ijkl,ijlm->ijkm', softmax_result, sliced_tensor).squeeze()
        # [batch,seq_number, column,pred_len]
        self.attn_weights = self.attn_weights.unsqueeze(-2)
        self.cnn_weights = torch.transpose(self.cnn_weights, 1, 2)
        # result[batch,seq_number,pred_len]
        result = torch.einsum('ijkl,ijlm->ijkm', self.attn_weights, self.cnn_weights).squeeze()
        result = self.Linear(result)
        result = result + y1
        result = self.Linear(result)

        return result
