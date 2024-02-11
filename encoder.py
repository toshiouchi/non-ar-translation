# -*- coding: utf-8 -*-

#
# Transformer エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from attention import ResidualAttentionBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(
        self,
        enc_vocab,
        enc_hidden_dim = 512,
        num_enc_layers = 6,
        enc_num_heads = 4,
        enc_kernel_size = [5,1],
        enc_filter_size = 2048,
        enc_input_maxlen = 3000,
        enc_dropout_rate = 0.1,
        dim_out = 2680
    ):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding( enc_vocab, enc_hidden_dim )
        self.pos_emb = nn.Embedding(enc_input_maxlen, enc_hidden_dim)
        # Attention Block
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(enc_hidden_dim, enc_num_heads, cross_attention = False ) for _ in range(num_enc_layers)]
        )
        self.dropout = nn.Dropout(p=enc_dropout_rate)
        self.input_maxlen = enc_input_maxlen
        
    def forward(self, x, in_lens, enc_padding_mask ):

        # テキストベクトル化
        out = self.embed( x )
        #print("size of out:{}".format( out.size()) )
        
        # position embbeding
        maxlen = out.size()[1]
        positions = torch.arange(start=0, end=self.input_maxlen, step=1).to(torch.long).to(device)
        #print( "size of positions:{}".format( positions.size() ))
        positions = self.pos_emb(positions.to(device))[:maxlen,:]
        #print( "size of positions:{}".format( positions.size() ))
        x = out.to(device) + positions.to(device)
        #x = self.dropout( x )
        
        # attention block
        attention_weights = []
        for i, block in enumerate( self.blocks ):
            x, attn1, attn2 = block(x, x, enc_padding_mask)
            attention_weights.append( attn1 )
            attention_weights.append( attn2 )
        
       
        return x, attention_weights  # (batch_size, input_seq_len, d_model)
