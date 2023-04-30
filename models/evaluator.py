import mindspore as ms
import mindspore.nn as nn
'''import sys
sys.path.append('./MTL-AQA')'''
from opts import *


class MLP_block(nn.Cell):

    def __init__(self, output_dim):
        super(MLP_block, self).__init__()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(axis=-1)
        self.layer1 = nn.Dense(feature_dim, 256)
        self.layer2 = nn.Dense(256, 128)
        self.layer3 = nn.Dense(128, output_dim)

    def construct(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        output = self.softmax(self.layer3(x))
        return output


class Evaluator(nn.Cell):

    def __init__(self, output_dim, model_type='USDL', num_judges=None):
        super(Evaluator, self).__init__()

        self.model_type = model_type

        if model_type == 'USDL':
            self.evaluator = MLP_block(output_dim=output_dim)
        else:
            assert num_judges is not None, 'num_judges is required in MUSDL'
            self.evaluator = nn.CellList([MLP_block(output_dim=output_dim) for _ in range(num_judges)])

    def construct(self, feats_avg):  # data: NCTHW

        if self.model_type == 'USDL':
            probs = self.evaluator(feats_avg)  # Nxoutput_dim
        else:
            probs = [evaluator(feats_avg) for evaluator in self.evaluator]  # len=num_judges
        return probs

if __name__ == '__main__':
    from mindspore.common.initializer import One, Normal
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_swin_bb', type=int, help='whether to use swin as backbone', default=1)
    args = parser.parse_args()

    evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL')
    clip_feats = ms.Tensor(shape=(1, 540, 1024), dtype=ms.float32, init=Normal())
    probs = evaluator(clip_feats.mean(1))
    print(f'probs.shape: {probs.shape}')