import mindspore as ms
import mindspore.nn as nn

class Linear_For_Backbone(nn.Cell):
    def __init__(self, args):
        super(Linear_For_Backbone, self).__init__()
        if args.use_swin_bb:
            input_dim = 1536
        else:
            input_dim = 768

        self.linear = nn.Dense(input_dim, 1024)
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(self.linear(x))

if __name__ == '__main__':
    from mindspore.common.initializer import One, Normal
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_swin_bb', type=int, help='whether to use swin as backbone', default=1)
    args = parser.parse_args()


    linear_bp = Linear_For_Backbone(args)

    X = ms.Tensor(shape=(1, 540, 1536), dtype=ms.float32, init=Normal())
    print(f'X.shape: {X.shape}')

    Y = linear_bp(X)
    print(f'Y.shape: {Y.shape}')