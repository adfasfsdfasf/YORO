# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import time
import os
from torch.utils.data import DataLoader, ConcatDataset
from statistics import mean
from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, evaluation_previous_batch,global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools
from tqdm import tqdm
import sampler
warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')    
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)

        fileHandler = logging.FileHandler(os.path.join(save_path, 'visa.txt'), encoding='utf-8')
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger
def get_sampler(device, name="approx_greedy_coreset", percentage=0.1):
    if name == "identity":
        return sampler.IdentitySampler()
    elif name == "random":
        return sampler.RandomSampler(percentage)
    elif name == "greedy_coreset":
        return sampler.GreedyCoresetSampler(percentage, device)
    elif name == "approx_greedy_coreset":
        return sampler.ApproximateGreedyCoresetSampler(percentage, device)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _detach(features):
    return [x.detach().cpu().numpy() for x in features]#将特征从计算图中分离，并移动到 CPU 上，然后转换为 NumPy 数组。

def train(item_list):
    setup_seed(1)

    total_iters = 10000
    batch_size = 16
    image_size = 448
    crop_size = 392

    # image_size = 448
    # crop_size = 448

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)


    for ji in range(1, 2):
        if ji == 1:

            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 2:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 3:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 4:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 5:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 6:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 7:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        train_data_list = [] 
        #setup_seed(ji+11)
        print_fn('seed:{}'.format(ji+11))
        test_data_list = []
        train_dataloader_list = []
        for i, item in enumerate(item_list):

            #item = 'cable'
            train_path = os.path.join(args.data_path, item, 'train')
            test_path = os.path.join(args.data_path, item)
            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]
            test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
            incre_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=8)
            train_dataloader_list.append(incre_train_loader)
            train_data_list.append(train_data)
            test_data_list.append(test_data)
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                                    drop_last=True)
        test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)
                                for test_data in test_data_list]

        # encoder_name = 'dinov2reg_vit_small_14'
        encoder_name = 'dinov2reg_vit_base_14'
        # encoder_name = 'dinov2reg_vit_large_14'

        # encoder_name = 'dinov2_vit_base_14'
        # encoder_name = 'dino_vit_base_16'
        # encoder_name = 'ibot_vit_base_16'
        # encoder_name = 'mae_vit_base_16'
        # encoder_name = 'beitv2_vit_base_16'
        # encoder_name = 'beit_vit_base_16'
        # encoder_name = 'digpt_vit_base_16'
        # encoder_name = 'deit_vit_base_16'

        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        # fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        # fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_encoder = [[0, 1, 2, 3,4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
        # target_layers = list(range(4, 19))

        encoder = vit_encoder.load(encoder_name)
        fps_mean_list = []

        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."

        bottleneck = []
        decoder = []

        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
        # bottleneck.append(nn.Sequential(FeatureJitter(scale=40),
        #                                 bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)))

        bottleneck = nn.ModuleList(bottleneck)

        for i in range(8):
            # blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
            #             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
            #             attn=LinearAttention2)#使用VitBlock构建标准transformer块。
            blk = ConvBlock(dim=embed_dim, kernel_size=7, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            # blk = SetTransformer(
            #     dim_input=embed_dim,
            #     num_heads=4,
            #     num_outputs=784,
            #     dim_output=embed_dim,
            # )
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)

        model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                    mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
        model = model.to(device)
        k = 11
        memory_features_list = torch.load('E:/model/EmbeddingAD-main/VISA2__results/VISA/11_VISA_memory_features_list4.pth').to(device)
        model.load_state_dict(torch.load('E:/model/EmbeddingAD-main/VISA2__results/VISA/12_model4.pth'))
        auroc_sp_list = []
        ap_sp_list = []
        f1_sp_list = []
        auroc_px_list = []
        ap_px_list = []
        f1_px_list = []
        aupro_px_list = []

        for test_i, test_data in enumerate(test_data_list):
            # if epoch < 2 and test_i > k:
            #     break
            if k < len(test_data_list)-1:
                break
            if test_i > k:
                break
            item = item_list[test_i]
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                            num_workers=4)
            if test_i < k:
                results = evaluation_previous_batch(model, test_dataloader, device, memory_features_list, max_ratio=0.01, resize_mask=256)

            elif test_i == k:
                 results= evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            
            print_fn(
                '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
        # item = item_list[k]
        # test_data = test_data_list[k]
        # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
        #                                                     num_workers=8)    
        # results, fps_list, fps_mean = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        # print_fn(fps_list)
        # print_fn(f'fps均值:{fps_mean}')
        # auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        # print_fn(
        #     '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        #             item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px)) 
        print_fn(
            'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    import argparse

    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--data_path', type=str, default='E:/Dataset/mvtec2d2/')
    #parser.add_argument('--data_path', type=str, default='E:/Dataset/VISA/')
    parser.add_argument('--data_path', type=str, default='E:/model/spot-diff-main/utils/VisA_pytorch/1cls')
    parser.add_argument('--save_dir', type=str, default='./evaluate_results')
    parser.add_argument('--save_name', type=str,
                        default='visa')
    args = parser.parse_args()
    #
    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper']
    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
              'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    print(torch.cuda.device_count())
    train(item_list)
