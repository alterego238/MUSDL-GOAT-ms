import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import math
from mindspore.common.initializer import initializer, HeNormal

class Attention(nn.Cell):
    def __init__(self, dim, linear_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.linear_dim = linear_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # QKV matrix
        self.q_matrix = nn.Dense(linear_dim, linear_dim, has_bias=qkv_bias)
        self.k_matrix = nn.Dense(linear_dim, linear_dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.bn = nn.BatchNorm1d(540, eps=1e-05, momentum=0.1, affine=True, use_batch_statistics=True)

        self.relu = nn.ReLU()
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        for m in self.cells():
            if isinstance(m, nn.Dense):
                for name, param in m.parameters_and_names():
                    if 'weight' in name:
                        param.set_data(initializer(HeNormal(), param.shape, param.dtype))
                    if 'bias' in name:
                        param.set_data(initializer('zeros', param.shape, param.dtype))

    def construct(self, q_in, k_in, x):
        B, N, C = x.shape
        q = self.q_matrix(q_in).reshape(B, N, self.num_heads, self.linear_dim // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'
        k = self.k_matrix(k_in).reshape(B, N, self.num_heads, self.linear_dim // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'

        attn = (q @ k.transpose((0, 1, 3, 2))) * self.scale  # B,num_heads,N,N
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,num_heads,N,C'
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B,N,C
        x = x + (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)  # B,N,C
        x = self.bn(x.permute(0, 2, 1).reshape(B * C, N)).reshape(B, C, N).permute(0, 2, 1).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        q = q.permute(0, 2, 1, 3).reshape(B, N, self.linear_dim)
        k = k.permute(0, 2, 1, 3).reshape(B, N, self.linear_dim)
        return q, k, x, attn


class Encoder_Blocks(nn.Cell):
    def __init__(self, qk_dim, dim, linear_dim, num_heads, num_layers, attn_drop=0., proj_drop=0.):
        super(Encoder_Blocks, self).__init__()
        model_list = []
        for i in range(num_layers):
            model_list.append(Attention(dim, linear_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop))
        self.model = nn.CellList(model_list)
        self.linear_q = nn.Dense(qk_dim, linear_dim)
        self.linear_k = nn.Dense(qk_dim, linear_dim)
        self.qk_dim = qk_dim

    def construct(self, q, k, x):
        attn_qk = 0
        q = self.linear_q(q)
        k = self.linear_k(k)
        for i, _layer in enumerate(self.model):
            q, k, x, attn = _layer(q, k, x)
            if i == 3:
                attn_qk = attn
        return x, attn_qk


def temporal_position_encoding(size):
    bs = size[0]
    max_len = size[1]
    d_model = size[2]
    pe = ms.Tensor(np.zeros((max_len, d_model)))
    position = ms.Tensor(np.arange(0, max_len)).unsqueeze(1)
    div_term = ops.exp(np.arange(0, d_model, 2) *
                       -(math.log(10000.0) / d_model))
    pe[:, 0::2] = ops.sin(position * div_term)
    pe[:, 1::2] = ops.cos(position * div_term)
    pe = pe.unsqueeze(0)
    pe_b = ops.concat([pe for i in range(bs)])
    return pe_b

if __name__ == '__main__':
    from mindspore.common.initializer import One, Normal
    import argparse

    '''from mindspore import context
    context.set_context(device_target='CPU')'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int, default=8, help='number of self-attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--linear_dim', type=int, default=1024, help='dimension of query and key')
    parser.add_argument('--attn_drop', type=float, default=0., help='drop prob of attention layer')
    parser.add_argument('--use_bp', type=int, help='whether to use bridge prompt features', default=0)
    args = parser.parse_args()
    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024

    attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
    clip_feats = ms.Tensor(shape=(1, 540, 1024), dtype=ms.float32, init=Normal())
    q = ms.Tensor(shape=(1, 540, 1024), dtype=ms.float32, init=Normal())
    k = q
    output = attn_encoder(q, k, clip_feats)
    clip_feats = output[0]
    attn = output[1]
    print(f'clip_feats.shape: {clip_feats.shape}')
    print(f'attn.shape: {attn.shape}')