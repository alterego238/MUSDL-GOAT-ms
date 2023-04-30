import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np

#from models.backbone import *
#from utils import *
#from roi_align.roi_align import RoIAlign  # RoIAlign module
from mindspore.common.initializer import initializer, HeNormal

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args:
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(axis=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(axis=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(0, 2, 1)) + ry.transpose(0, 2, 1)

    return ops.sqrt(dist)

class GCN_Module(nn.Cell):
    def __init__(self, args):
        super(GCN_Module, self).__init__()
        self.args = args

        # dimension and layers number
        NFR = args.num_features_relation  # dimension of features of phi(x) and theta(x) in 'Embedded Dot-Product'
        NG = args.num_graph
        NFG = args.num_features_gcn  # dimension of features of node in graphs
        NFG_ONE = NFG

        # set modules
        self.fc_rn_theta_list = nn.CellList([nn.Dense(NFG, NFR) for i in range(NG)])
        self.fc_rn_phi_list = nn.CellList([nn.Dense(NFG, NFR) for i in range(NG)])
        self.fc_gcn_list = nn.CellList([nn.Dense(NFG, NFG_ONE, has_bias=False) for i in range(NG)])
        self.nl_gcn_list = nn.CellList([nn.LayerNorm([NFG_ONE]) for i in range(NG)])

    def construct(self, graph_boxes_features, boxes_in_flat):
        """
        graph_boxes_features  [B*T,N,NFG]
        """

        # GCN graph modeling
        # Prepare boxes similarity relation
        B, N, NFG = graph_boxes_features.shape  # Note! B is actually B*T
        NFR = self.args.num_features_relation
        NG = self.args.num_graph
        NFG_ONE = NFG
        OH, OW = self.args.out_size
        pos_threshold = self.args.pos_threshold

        # Prepare position mask
        graph_boxes_positions = boxes_in_flat  # B*T*N, 4
        graph_boxes_positions[:, 0] = (graph_boxes_positions[:, 0] + graph_boxes_positions[:, 2]) / 2
        graph_boxes_positions[:, 1] = (graph_boxes_positions[:, 1] + graph_boxes_positions[:, 3]) / 2
        graph_boxes_positions = graph_boxes_positions[:, :2].reshape(B, N, 2)  # B*T, N, 2
        graph_boxes_distances = calc_pairwise_distance_3d(graph_boxes_positions, graph_boxes_positions)  # B(*T)?, N, N
        position_mask = (graph_boxes_distances > (pos_threshold * OW))

        # GCN list
        relation_graph = None
        graph_boxes_features_list = []
        for i in range(NG):
            # calculate similarity
            graph_boxes_features_theta = self.fc_rn_theta_list[i](graph_boxes_features)  # B*T,N,NFR
            graph_boxes_features_phi = self.fc_rn_phi_list[i](graph_boxes_features)  # B*T,N,NFR
            similarity_relation_graph = ops.matmul(graph_boxes_features_theta,
                                                     graph_boxes_features_phi.transpose(0, 2, 1))  # B*T,N,N
            similarity_relation_graph = similarity_relation_graph / np.sqrt(NFR)
            similarity_relation_graph = similarity_relation_graph.reshape(-1, 1)  # B*T*N*N, 1

            # Build relation graph
            relation_graph = similarity_relation_graph
            relation_graph = relation_graph.reshape(B, N, N)  # B*T,N,N
            relation_graph[position_mask] = -float('inf')
            relation_graph = ops.softmax(relation_graph, axis=2)

            # Graph convolution
            one_graph_boxes_features = self.fc_gcn_list[i](
                ops.matmul(relation_graph, graph_boxes_features))  # B, N, NFG_ONE  # G*X*W
            one_graph_boxes_features = self.nl_gcn_list[i](one_graph_boxes_features)
            one_graph_boxes_features = ops.relu(one_graph_boxes_features)  # ReLu(G*X*W)
            graph_boxes_features_list.append(one_graph_boxes_features)

        # fuse multi graphs
        graph_boxes_features = ops.stack(graph_boxes_features_list).sum(axis=0)  # B*T, N, NFG

        return graph_boxes_features, relation_graph


class GCNnet_artisticswimming_simplified(nn.Cell):
    """
    main module of GCN for the volleyball dataset
    """

    def __init__(self, args):
        super(GCNnet_artisticswimming_simplified, self).__init__()
        self.args = args

        # set parameters
        N = self.args.num_boxes
        D = self.args.emb_features  # output feature map channel of backbone
        K = self.args.crop_size[0]  # crop size of roi align
        NFB = self.args.num_features_boxes
        NFR, NFG = self.args.num_features_relation, self.args.num_features_gcn

        # set modules
        self.gcn_list = nn.CellList([GCN_Module(args) for i in range(self.args.gcn_layers)])

        # initial
        for m in self.cells():
            if isinstance(m, nn.Dense):
                for name, param in m.parameters_and_names():
                    if 'weight' in name:
                        param.set_data(initializer(HeNormal(), param.shape, param.dtype))
                    if 'bias' in name:
                        param.set_data(initializer('zeros', param.shape, param.dtype))

    def construct(self, boxes_features, boxes_in):

        # read config parameters
        B = boxes_features.shape[0]  # B,540*t,N,NFB
        T = boxes_features.shape[1]
        t = self.args.num_selected_frames
        N = self.args.num_boxes
        NFB = self.args.num_features_boxes
        NFR, NFG = self.args.num_features_relation, self.args.num_features_gcn

        boxes_in_flat = ops.reshape(boxes_in, (B * T * N, 4))  # B*T*N, 4
        boxes_in_flat.requires_grad = False
        # GCN
        if self.args.use_gcn:
            if self.args.gcn_temporal_fuse:
                graph_boxes_features = boxes_features.reshape(B, T * N, NFG)
            else:
                graph_boxes_features = boxes_features.reshape(B * T, N, NFG)
            for i in range(len(self.gcn_list)):
                graph_boxes_features, relation_graph = self.gcn_list[i](graph_boxes_features, boxes_in_flat)
        else:
            graph_boxes_features = boxes_features

        # fuse graph_boxes_features with boxes_features
        graph_boxes_features = graph_boxes_features.reshape(B, T, N, NFG)
        boxes_features = boxes_features.reshape(B, T, N, NFB)
        boxes_states = graph_boxes_features + boxes_features

        # reshape as query
        boxes_states = boxes_states.reshape(B, self.args.length // 10, t, N, NFG)
        boxes_states = boxes_states.mean(2).mean(2)  # B,540,1024

        return boxes_states

if __name__ == '__main__':
    from mindspore.common.initializer import One, Normal
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_boxes', type=int, help='boxes number of each frames', default=8)
    parser.add_argument('--num_selected_frames', type=int, help='number of selected frames per 16 frames', default=1)
    parser.add_argument('--num_features_boxes', type=int, default=1024, help='dimension of features of each box')
    parser.add_argument('--num_features_relation', type=int, default=256, help='dimension of embedding phi(x) and theta(x) [Embedded Dot-Product]')
    parser.add_argument('--num_features_gcn', type=int, default=1024, help='dimension of features of each node')
    parser.add_argument('--use_gcn', type=int, help='whether to use gcn', default=1)
    parser.add_argument('--gcn_temporal_fuse', type=int, help='whether to fuse temporal node before gcn', default=0)
    parser.add_argument('--num_graph', type=int, default=16, help='number of graphs')
    parser.add_argument('--gcn_layers', type=int, default=1, help='number of gcn layers')
    parser.add_argument('--pos_threshold', type=float, default=0.2, help='threshold for distance mask')
    parser.add_argument('--out_size', type=tuple, help='output image size', default=(25, 25))
    parser.add_argument('--crop_size', type=tuple, help='RoiAlign image size', default=(5, 5))
    parser.add_argument('--emb_features', type=int, default=1056, help='output feature map channel of backbone')
    parser.add_argument('--length', type=int, help='length of videos', default=5406)
    args = parser.parse_args()

    gcn = GCNnet_artisticswimming_simplified(args)
    boxes_features = ms.Tensor(shape=(1, 540, 8, 1024), dtype=ms.float32, init=Normal())
    boxes_in = ms.Tensor(shape=(1, 540, 8, 4), dtype=ms.float32, init=Normal()) # B,T,N,4
    q = gcn(boxes_features, boxes_in)  # B,540,1024
    print(f"q.shape: {q.shape}")