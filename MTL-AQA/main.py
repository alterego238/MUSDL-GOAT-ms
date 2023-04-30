import os
import pickle
import sys
#from thop import profile

sys.path.append('../')

import mindspore as ms
import mindspore.nn as nn

from mindspore.dataset import GeneratorDataset
from utils import *
from opts import *
from scipy import stats
from tqdm import tqdm
from dataset import VideoDataset
#from models.i3d import InceptionI3d
from models.evaluator import Evaluator
from config import get_parser
import time
#from models.cnn_model import GCNnet_artisticswimming
from models.cnn_simplified import GCNnet_artisticswimming_simplified
from models.group_aware_attention import Encoder_Blocks
from utils import *
from models.linear_for_bp import Linear_For_Backbone

'''torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True'''


def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    # i3d = InceptionI3d()
    # i3d.load_state_dict(torch.load(i3d_pretrained_path))
    i3d = 0

    if args.type == 'USDL':
        evaluator = Evaluator(output_dim=output_dim['USDL'], model_type='USDL')
    else:
        evaluator = Evaluator(output_dim=output_dim['MUSDL'], model_type='MUSDL', num_judges=num_judges)

    # if len(args.gpu.split(',')) > 1:
    #     # i3d = nn.DataParallel(i3d)
    #     evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator


def compute_score(model_type, probs, data):
    if model_type == 'USDL':
        pred = probs.argmax(axis=-1) * (label_max / (output_dim['USDL'] - 1))
    else:
        # calculate expectation & denormalize & sort
        judge_scores_pred = ops.stack([prob.argmax(dim=-1) * judge_max / (output_dim['MUSDL'] - 1)
                                         for prob in probs], axis=1).sort()[0]  # N, 7

        # keep the median 3 scores to get final score according to the rule of diving
        pred = ops.sum(judge_scores_pred[:, 2:5], axis=1) * data['difficulty']
    return pred


def compute_loss(model_type, criterion, probs, data):
    if model_type == 'USDL':
        loss = criterion(ops.log(probs), data['soft_label'])
    else:
        loss = sum([criterion(ops.log(probs[i]), data['soft_judge_scores'][:, i]) for i in range(num_judges)])
    return loss


def get_dataloaders(args):
    dataloaders = {}
    test_dataset_generator = VideoDataset('test', args)
    test_dataset = GeneratorDataset(test_dataset_generator, ["data"], shuffle=False, num_parallel_workers=args.num_workers)
    test_dataset = test_dataset.batch(batch_size=args.test_batch_size)
    test_dataloader = test_dataset.create_tuple_iterator()
    '''dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)'''

    if args.use_multi_gpu:
        raise NotImplementedError()
        dataloaders['train'] = build_dataloader(VideoDataset('train', args),
                                                batch_size=args.train_batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                persistent_workers=True,
                                                seed=set_seed(args.seed))
    else:
        train_dataset_generator = VideoDataset('train', args)
        train_dataset = GeneratorDataset(train_dataset_generator, ["data"], shuffle=False, num_parallel_workers=args.num_workers)
        train_dataset = train_dataset.batch(batch_size=args.train_batch_size)
        train_dataloader = train_dataset.create_tuple_iterator()
        '''dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                           batch_size=args.train_batch_size,
                                                           num_workers=args.num_workers,
                                                           shuffle=False,
                                                           pin_memory=False,
                                                           worker_init_fn=worker_init_fn)'''
    return dataloaders


'''def flops_params(model, model_name: str, input_size: tuple):
    input = torch.randn(*input_size)
    flops, params = profile(model, inputs=(input, ))
    print(f'[{model_name}]flops: ', flops, 'params: ', params)'''


def main(dataloaders, i3d, evaluator, base_logger, args):
    # Print configuration
    '''if is_main_process():
        print('=' * 40)
        for k, v in vars(args).items():
            print(f'{k}: {v}')
        print('=' * 40)'''
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    if args.use_bp:
        args.qk_dim = 768
    else:
        args.qk_dim = 1024

    # Set loss function
    criterion = nn.KLDivLoss()

    '''# Set data position
    if torch.cuda.is_available():
        device = get_device()
    else:
        device = torch.device('cpu')'''

    # Set models and optimizer(depend on whether to use goat)
    if args.use_goat:
        if args.use_cnn_features:
            gcn = GCNnet_artisticswimming_simplified(args)

            '''input1 = torch.randn(1, 540, 8, 1024)
            input2 = torch.randn(1, 540, 8, 4)
            flops, params = profile(gcn, inputs=(input1, input2))
            print(f'[GCNnet_artisticswimming_simplified]flops: ', flops, 'params: ', params)'''
        else:
            raise NotImplementedError()
            gcn = GCNnet_artisticswimming(args)
            gcn.loadmodel(args.stage1_model_path)
        attn_encoder = Encoder_Blocks(args.qk_dim, 1024, args.linear_dim, args.num_heads, args.num_layers, args.attn_drop)
        linear_bp = Linear_For_Backbone(args)

        '''input1 = torch.randn(1, 540, 1024)
        input2 = torch.randn(1, 540, 1024)
        input3 = torch.randn(1, 540, 1024)
        flops, params = profile(attn_encoder, inputs=(input1, input2, input3))
        print(f'[attn_encoder]flops: ', flops, 'params: ', params)

        flops, params = profile(evaluator, inputs=(input1.mean(1), ))
        print(f'[evaluator]flops: ', flops, 'params: ', params)'''

        if args.use_multi_gpu:
            raise NotImplementedError()
            wrap_model(gcn, distributed=args.distributed)
            wrap_model(attn_encoder, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
            wrap_model(evaluator, distributed=args.distributed)
        else:
            gcn = gcn
            attn_encoder = attn_encoder
            linear_bp = linear_bp
            evaluator = evaluator
        optimizer = nn.Adam([
            {'params': gcn.trainable_params()},
            {'params': evaluator.trainable_params()},
            {'params': linear_bp.trainable_params()},
            {'params': attn_encoder.trainable_params()}
        ], learing_rate=args.lr, weight_decay=args.weight_decay)
    else:
        gcn = None
        attn_encoder = None
        linear_bp = Linear_For_Backbone(args)
        if args.use_multi_gpu:
            raise NotImplementedError()
            wrap_model(evaluator, distributed=args.distributed)
            wrap_model(linear_bp, distributed=args.distributed)
        else:
            evaluator = evaluator
            linear_bp = linear_bp
        optimizer = nn.Adam([{'params': evaluator.trainable_params()}, {'params': linear_bp.trainable_params()}], learing_rate=args.lr, weight_decay=args.weight_decay)

    # DP
    # evaluator = nn.DataParallel(evaluator)
    # if args.use_goat:
    #     gcn = nn.DataParallel(gcn)
    #     attn_encoder = nn.DataParallel(attn_encoder)

    # train or test
    epoch_best = 0
    rho_best = 0
    RL2_best = 100
    rho = -1
    RL2 = 100
    for epoch in range(args.num_epochs):
        if args.use_multi_gpu:
            raise NotImplementedError()
            dataloaders['train'].sampler.set_epoch(epoch)
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best rho: {rho_best} at epoch {epoch_best}, Current Best RL2: {RL2_best}')
        '''if is_main_process():
            log_and_print(base_logger, f'Epoch: {epoch}  Current Best rho: {rho_best} at epoch {epoch_best}, Current Best RL2: {RL2_best}')'''

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                # i3d.train()
                if args.use_goat:
                    gcn.set_train()
                    attn_encoder.set_train()
                evaluator.set_train()
                linear_bp.set_train()
                #torch.set_grad_enabled(True)
            else:
                # i3d.eval()
                if args.use_goat:
                    gcn.set_train(False)
                    attn_encoder.set_train(False)
                evaluator.set_train(False)
                linear_bp.set_train(False)
                #torch.set_grad_enabled(False)
            # visual
            attn_list = []
            key_list = []
            if split == 'train' or (split == 'test'):# and is_main_process()):
                for data in dataloaders[split]:
                    true_scores.extend(data['final_score'].asnumpy())
                    clip_feats = data['feature']  # B,540,1024
                    if not args.use_i3d_bb:
                        clip_feats = linear_bp(clip_feats)  # B,540,1024

                    if split == 'train':
                        def forward_fn():
                            start = time.time()
                            ######### GOAT START ##########
                            if args.use_goat:
                                # Use formation features
                                if args.use_formation:
                                    q = data['formation_features']  # B,540,1024
                                    k = q
                                    # clip_feats = attn_encoder(q, k, clip_feats)
                                    output = attn_encoder(q, k, clip_feats)
                                    clip_feats = output[0]
                                    attn = output[1]
                                # Use bridge-prompt features
                                elif args.use_bp:
                                    q = data['bp_features']  # B,540,768
                                    k = q
                                    # clip_feats = attn_encoder(q, k, clip_feats)
                                    output = attn_encoder(q, k, clip_feats)
                                    clip_feats = output[0]
                                    attn = output[1]
                                # Use self-attention
                                elif args.use_self:
                                    q = clip_feats
                                    k = q
                                    # clip_feats = attn_encoder(q, k, clip_feats)
                                    output = attn_encoder(q, k, clip_feats)
                                    clip_feats = output[0]
                                    attn = output[1]
                                # Use group features
                                else:
                                    if args.use_cnn_features:
                                        boxes_features = data['cnn_features']
                                        boxes_in = data['boxes']  # B,T,N,4
                                        q = gcn(boxes_features, boxes_in)  # B,540,1024
                                        k = q
                                        # clip_feats = attn_encoder(q, k, clip_feats)
                                        output = attn_encoder(q, k, clip_feats)
                                        clip_feats = output[0]
                                        attn = output[1]
                                    else:
                                        images_in = data['video']  # B,T,C,H,W
                                        boxes_in = data['boxes']  # B,T,N,4
                                        q = gcn(images_in, boxes_in)  # B,540,1024
                                        k = q
                                        # clip_feats = attn_encoder(q, k, clip_feats)
                                        output = attn_encoder(q, k, clip_feats)
                                        clip_feats = output[0]
                                        attn = output[1]
                                if split != 'train':
                                    attn_list.append(attn)
                                    key_list.append(data['key'])
                            #########  GOAT END  ##########

                            probs = evaluator(clip_feats.mean(1))
                            preds = compute_score(args.type, probs, data)
                            pred_scores.extend([i.item() for i in preds])
                            loss = compute_loss(args.type, criterion, probs, data)

                            infer_time = time.time() - start
                            return loss, pred_scores, infer_time
                        
                        grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                        (loss, pred_scores, infer_time), grads = grad_fn()
                        loss = ops.depend(loss, optimizer(grads))
                        '''loss = compute_loss(args.type, criterion, probs, data)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()'''
                    else:
                        start = time.time()
                        ######### GOAT START ##########
                        if args.use_goat:
                            # Use formation features
                            if args.use_formation:
                                q = data['formation_features']  # B,540,1024
                                k = q
                                # clip_feats = attn_encoder(q, k, clip_feats)
                                output = attn_encoder(q, k, clip_feats)
                                clip_feats = output[0]
                                attn = output[1]
                            # Use bridge-prompt features
                            elif args.use_bp:
                                q = data['bp_features']  # B,540,768
                                k = q
                                # clip_feats = attn_encoder(q, k, clip_feats)
                                output = attn_encoder(q, k, clip_feats)
                                clip_feats = output[0]
                                attn = output[1]
                            # Use self-attention
                            elif args.use_self:
                                q = clip_feats
                                k = q
                                # clip_feats = attn_encoder(q, k, clip_feats)
                                output = attn_encoder(q, k, clip_feats)
                                clip_feats = output[0]
                                attn = output[1]
                            # Use group features
                            else:
                                if args.use_cnn_features:
                                    boxes_features = data['cnn_features']
                                    boxes_in = data['boxes']  # B,T,N,4
                                    q = gcn(boxes_features, boxes_in)  # B,540,1024
                                    k = q
                                    # clip_feats = attn_encoder(q, k, clip_feats)
                                    output = attn_encoder(q, k, clip_feats)
                                    clip_feats = output[0]
                                    attn = output[1]
                                else:
                                    images_in = data['video']  # B,T,C,H,W
                                    boxes_in = data['boxes']  # B,T,N,4
                                    q = gcn(images_in, boxes_in)  # B,540,1024
                                    k = q
                                    # clip_feats = attn_encoder(q, k, clip_feats)
                                    output = attn_encoder(q, k, clip_feats)
                                    clip_feats = output[0]
                                    attn = output[1]
                            if split != 'train':
                                attn_list.append(attn)
                                key_list.append(data['key'])
                        #########  GOAT END  ##########

                        probs = evaluator(clip_feats.mean(1))
                        preds = compute_score(args.type, probs, data)
                        pred_scores.extend([i.item() for i in preds])

                        infer_time = time.time() - start

                rho, p = stats.spearmanr(pred_scores, true_scores)
                pred_scores = np.array(pred_scores)
                true_scores = np.array(true_scores)
                RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

                log_and_print(base_logger, f'epoch:{epoch}, {split} correlation: {rho}, Rl2: {RL2}, Infer_Time: {infer_time:.6f}')
                '''if is_main_process():
                    log_and_print(base_logger, f'epoch:{epoch}, {split} correlation: {rho}, Rl2: {RL2}, Infer_Time: {infer_time:.6f}')'''

        if rho > rho_best and split == 'test':# and is_main_process():
            if args.use_goat:
                attn_list_log = attn_list
                key_list_log = key_list
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '-----New best rho found!-----')
            '''if is_main_process():
                log_and_print(base_logger, '-----New best rho found!-----')'''
            if args.save:
                ms.save_checkpoint(append_dict={'epoch': epoch,
                            # 'i3d': i3d.state_dict(),
                            'evaluator': evaluator.parameters_dict(),
                            'optimizer': optimizer.parameters_dict(),
                            'rho_best': rho_best}, ckpt_file_name=f'ckpts/{args.type}.pt')
        if RL2 < RL2_best and split == 'test':# and is_main_process():
            RL2_best = RL2
            log_and_print(base_logger, '-----New best RL2 found!-----')
            '''if is_main_process():
                log_and_print(base_logger, '-----New best RL2 found!-----')'''
        if epoch == args.num_epochs - 1:# and is_main_process():
            log_best(rho_best, RL2_best, epoch_best, args)
            visual_dict = {'attn': attn_list_log, 'key': key_list_log}
            dict_root = f'attn_visual/{rho_best:.4f}_attention_visualization.pkl'
            pickle.dump(visual_dict, open(dict_root, 'wb'))


if __name__ == '__main__':

    args = get_parser()

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True

    # setup_env(args.launcher, distributed=args.distributed)

    init_seed(args)

    localtime = time.asctime(time.localtime(time.time()))
    base_logger = get_logger(f'exp/{args.type}_full_split{args.split}_{args.lr}_{localtime}.log', args.log_info)
    i3d, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, base_logger, args)
