import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_d = dim // self.num_heads
        self.dim = dim
        self.attn_drop = attn_drop
        self.scale = self.head_d ** -0.5 #縮放因子，用來在計算attention得分時對結果進行縮放，防止數值過大。


        self.projection = nn.Linear(self.dim, self.dim)
        self.attn_drop = nn.Dropout(self.attn_drop)
        self.transfer_qkv = nn.Linear(self.dim , self.num_heads * self.head_d * 3 , bias = False)
    
    def forward(self, x):
        batch_size, num, dimension = x.shape

        # 生成 q, k, v 並拆分為多頭形式
        qkv = self.transfer_qkv(x)
        qkv = qkv.view(batch_size, num, 3, self.num_heads, self.head_d)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # 計算attention score
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        # 計算attention output
        context_layer = (attention_probs @ v).permute(1, 2, 0, 3)
        context_layer = context_layer.reshape(batch_size, num, dimension)
        output = self.projection(context_layer)

        return output

        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        #raise Exception('TODO1!')

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    