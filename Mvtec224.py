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

from models.uad import ViTill, ViTillv2, ViTill
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, evaluation_previous_batch,global_cosine, calculate_means, visualize,regional_cosine_hm_percent, \
    global_cosine_hm_percent, \
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
import image_sampler
warnings.filterwarnings("ignore")
def fuse_feature(feat_list):
    return torch.stack(feat_list, dim=1).mean(dim=1)
def getKDprediction(x, bottleneck, decoder):
    fuse_layer_decoder = [[0, 1, 2, 3, 4, 5, 6, 7]]
    for i, blk in enumerate(bottleneck):
        x = blk(x)
    de_list = []
    for i, blk in enumerate(decoder):
        x = blk(x, attn_mask=None)
        de_list.append(x)
    de_list = de_list[::-1]
    de = [fuse_feature([de_list[idx] for idx in idxs]) for idxs in fuse_layer_decoder]
    #de = [d[:, 5:, :] for d in de]
    return de[0] 

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')    
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)

        fileHandler = logging.FileHandler(os.path.join(save_path, 'final_mvtec.txt'), encoding='utf-8')
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
    elif name == "kcenter":
        return sampler.KMeansPlusPlusSampler(percentage, device)

def get_image_sampler(device, name="approx_greedy_coreset", percentage=0.5):
    if name == "identity":
        return image_sampler.IdentitySampler()
    elif name == "random":
        return image_sampler.RandomSampler(percentage)
    elif name == "greedy_coreset":
        return image_sampler.GreedyCoresetSampler(percentage, device)
    elif name == "approx_greedy_coreset":
        return image_sampler.ApproximateGreedyCoresetSampler(percentage, device)
    elif name == "kcenter":
        return image_sampler.KMeansPlusPlusSampler(percentage, device)

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
    crop_size = 224

    # image_size = 448
    # crop_size = 448

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)


    for ji in range(4, 5):
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
        elif ji == 8:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 9:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 10:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        elif ji == 11:
            featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=784)
        featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=768)
        image_sampler = get_image_sampler(device, name="approx_greedy_coreset", percentage=0.5)
        lamda = 1 - 8 * 0.05
        beta = 0 + 0.025 * ji
        lam = np.random.beta(lamda, lamda)

        print_fn('lamda:{}'.format(lamda))
        print_fn('lam:{}'.format(lam))
        print_fn('beta:{}'.format(beta))
        print_fn('ji:{}'.format(ji+1))
        train_data_list = []
        FM_metric_list = np.zeros((len(item_list), 2))  # [I_AUROC, P_AP]
        final_metric_list = np.zeros((len(item_list), 2))  # [I_AUROC, P_AP]
        #setup_seed(ji)
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
            incre_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)
            train_dataloader_list.append(incre_train_loader)
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                    drop_last=True)
        test_dataloader_list = [torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
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
        KD_bottleneck = copy.deepcopy(bottleneck).to(device)
        convex_bottleneck = copy.deepcopy(bottleneck).to(device)

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
        KD_decoder= copy.deepcopy(decoder).to(device)
        convex_decoder = copy.deepcopy(decoder).to(device)


        model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                    mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
        model = model.to(device)
        trainable = nn.ModuleList([bottleneck, decoder])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW([{'params': trainable.parameters()}],
                                lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                        warmup_iters=100)
        #print_fn('train image number:{}'.format(len(train_data)))
        #featuresampler = get_sampler(device, name="approx_greedy_coreset", percentage=0.1)
        it = 0
        k = 0
        memory_features_list = None
        for train_loader in train_dataloader_list:
            memory_features = None
            #train_data = ConcatDataset(train_data_list[:k + 1])
            #print_fn('train image number:{}'.format(len(train_data_list[k])))
            #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
            #for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
            epochs = tqdm(range(5), desc="Training Epochs", total=5)
            if k > 0:
                current_task_en_list = []
                progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
                model.eval()
                current_batch_array2_all = None
                # 计算当前任务的特征,并保存记忆特征
                for _, (img, label) in progress_bar:    
                    img = img.to(device)
                    en_list = model(img, train=1)
                    current_batch_features = _detach(en_list.reshape(-1, 768).contiguous())
                    current_batch_features2 = _detach(en_list.reshape(-1, 256*768).contiguous())
                    current_batch_array = np.stack(current_batch_features)
                    current_batch_array2 = np.stack(current_batch_features2)
                    if current_batch_array2_all is None:
                        current_batch_array2_all = current_batch_array2
                    else:
                        current_batch_array2_all = np.concatenate([current_batch_array2_all, current_batch_array2], axis=0)
                    if memory_features is None:
                        memory_features = current_batch_array
                    else:
                        memory_features = np.concatenate([memory_features, current_batch_array], axis=0)
                    for en in en_list:
                        current_task_en_list.append(en)
                current_batch_array2_all = image_sampler.run(current_batch_array2_all)
                current_task_en_list = torch.from_numpy(current_batch_array2_all).view(-1, 256, 768).contiguous().cpu()
                #current_task_en_list = torch.stack(current_task_en_list, dim=0).cpu()  # 堆叠为 [len(en_list), batchsize, 789, 768]

                all_task_en = torch.cat([memory_features_list, current_task_en_list], dim=0) if memory_features_list is not None else current_task_en_list
                print_fn(f"Memory features shape: {memory_features_list.shape}")
                print_fn(f"Current task features shape: {current_task_en_list.shape}")
                print_fn(f"All task features shape: {all_task_en.shape}")
                #将当前任务的特征随机划分为多个batch进行训练
                perm = torch.randperm(all_task_en.size(0))  # 随机打乱顺序
                shuffled = all_task_en[perm]  # 根据打乱的顺序
                batches = [shuffled[i:i + batch_size] for i in range(0, shuffled.size(0), batch_size)]
                print_fn(f"Number of batches: {len(batches)}")
                train_loader = DataLoader(batches, batch_size=None, shuffle=True, num_workers=4)
                KD_batches = memory_features_list.to(device)  # 将记忆特征转换为张量并移动到设备上
                    
                for epoch in epochs:
                    model.train()             
                    KD_bottleneck.eval()
                    KD_decoder.eval()
                    convex_bottleneck.eval()
                    convex_decoder.eval()
                    loss_list = []   
                    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
                    for _, en in progress_bar:
                        en = en.to(device)#torch.Size([16, 784, 768])
                        en1, de = model(en, train=2)#torch.Size([16, 768, 28, 28])
                        p_final = 0.9
                        p = min(p_final * it / 1000, p_final)

                        #loss = global_cosine_hm_percent(en1, de, p=p, factor=0.1) + beta * min_loss
                        loss = global_cosine(en1, de) # + beta * min_loss
                         #+ beta * min_loss
                        # loss = global_cosine(en, de)
                        start_time = time.time()
                        optimizer.zero_grad()
                        loss.backward(retain_graph=False)
                        ##print(f"Gradient clipping time: {time.time() - start_time:.4f} seconds")
                        start_time = time.time()
                        optimizer.step()
                        #print(f"Optimizer step time: {time.time() - start_time:.4f} seconds")
                        loss_list.append(loss.item())
                        lr_scheduler.step()            
                        it += 1
                        if it == total_iters:
                            break 
                         
                torch.cuda.empty_cache()
                # if k ==1 or k == 9: 
                KD_bottleneck.load_state_dict(bottleneck.state_dict())
                KD_decoder.load_state_dict(decoder.state_dict())
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
                print_fn(f"以下是第{k + 1}个任务的结果：")
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
                        #visualize_previous(model, test_dataloader, memory_features_list, device, _class_=item, save_name=args.save_name)
                        results = evaluation_previous_batch(model, test_dataloader, device, memory_features_list, max_ratio=0.01, resize_mask=256)

                    if test_i == k:
                        #visualize(model, test_dataloader, device, _class_=item, save_name=args.save_name)
                        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)

                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    if auroc_sp > FM_metric_list[test_i, 0]:
                        FM_metric_list[test_i, 0] = auroc_sp
                    
                    if ap_px > FM_metric_list[test_i, 1]:
                         FM_metric_list[test_i, 1] = ap_px
                    if k == len(test_data_list) - 1:
                         final_metric_list[test_i, 0] = auroc_sp
                         final_metric_list[test_i, 1] = ap_px
                    if k == len(test_data_list) - 1 and test_i == k:
                         mean_dim1, mean_dim2 = calculate_means(FM_metric_list, final_metric_list)
                         print_fn(f"最高值的FM：I_AUROC {FM_metric_list[:, 0]} P_AP：{FM_metric_list[:, 1]}")
                         print_fn(f"最后的FM：I_AUROC {final_metric_list[:, 0]} P_AP：{final_metric_list[:, 1]}")
                         print_fn(f"最终结果的FM：I_AUROC {mean_dim1} P_AP：{mean_dim2}")

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
                # if k < len(item_list) - 1:
                #     item = item_list[k]
                #     test_data = test_data_list[k]
                #     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                #                                                         num_workers=8)    
                #     results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                #     auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                #     print_fn(
                #         '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                #                 item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px)) 
                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train()
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))


            elif k == 0:    
                for epoch in epochs:
                    model.train()

                    loss_list = []

                    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
                    for _, (img, label) in progress_bar:
                        img = img.to(device)#torch.Size([16, 3, 392, 392])
                        label = label.to(device)

                        en, de, en_list = model(img, train=0)
                        if epoch == 0:
                            current_batch_features = _detach(en_list.reshape(-1, 768).contiguous())
                            current_batch_array = np.stack(current_batch_features)
                            if memory_features is None:
                                memory_features = current_batch_array
                            else:
                                memory_features = np.concatenate([memory_features, current_batch_array], axis=0)
                        # loss = global_cosine(en, de)

                        p_final = 0.9
                        p = min(p_final * it / 1000, p_final)
                        #loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)
                        loss = global_cosine(en, de)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                        optimizer.step()
                        loss_list.append(loss.item())
                        lr_scheduler.step()            
                        it += 1
                        if it == total_iters:
                            break
                # if (it + 1) % 1 == 0:
                # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
                KD_bottleneck.load_state_dict(bottleneck.state_dict())
                KD_decoder.load_state_dict(decoder.state_dict())
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for test_i, test_data in enumerate(test_data_list):
                    if epoch < 2 and test_i > k:
                        break
                    if test_i > k:
                        break
                    item = item_list[test_i]
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                    num_workers=4)
                    if test_i == k:
                        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    if auroc_sp > FM_metric_list[test_i, 0]:
                        FM_metric_list[test_i, 0] = auroc_sp
                    if ap_px > FM_metric_list[test_i, 1]:
                        FM_metric_list[test_i, 1] = ap_px
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(f"以下是第{k + 1}个任务的结果：")
                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))           
                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train()
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
            k += 1
            start_time = time.time()
            #np.save('memory_features.npy', memory_features)
            memory_features = featuresampler.run(memory_features)
            memory_features = torch.from_numpy(memory_features)
            memory_features = memory_features.view(-1, 256, 768).contiguous()  # 将内存特征转换为PyTorch张量，并移动到指定的设备上。
            #torch.save(memory_features, os.path.join(args.save_dir, args.save_name, f'visa_memory_features_{k}.pth'))
            print_fn(f"当前任务记忆特征shape: {memory_features.shape}")
            #torch.save(memory_features, os.path.join(args.save_dir, args.save_name, f'memory_features_{k}.pth'))
            print_fn(f"当前任务记忆特征shape: {memory_features.shape}")
            #memory_features_list = np.concatenate([memory_features_list, memory_features], axis=0) if memory_features_list is not None else memory_features
            #memory_features_list = [torch.from_numpy(x).view(789, 768).contiguous() for x in memory_features_list]#将内存特征转换为PyTorch张量，并移动到指定的设备上。
            #memory_features_list = torch.stack(memory_features_list, dim=0).cpu()  # 堆叠为 [len, 789, 768]
            if memory_features_list is None:
                memory_features_list = memory_features
            else:
                memory_features_list = torch.cat([memory_features_list, memory_features], dim=0).cpu()  # 堆叠为 [len, 789, 768]
            end_time = time.time()
            print_fn(f"合并先前记忆特征之后的shape: {memory_features_list.shape}")
            print_fn(f"Feature sampling time: {end_time - start_time:.4f}s")
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'{k}_model4.pth'))
            torch.save(memory_features_list, os.path.join(args.save_dir, args.save_name, f'{k}_VISA_memory_features_list4.pth'))
    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='E:/Dataset/mvtec2d2/')
    #parser.add_argument('--data_path', type=str, default='E:/Dataset/Visa/')
    #parser.add_argument('--data_path', type=str, default='E:/model/spot-diff-main/utils/VisA_pytorch/1cls')
    #parser.add_argument('--data_path', type=str, default='E:/Dataset/MVTec_VISA/')

    parser.add_argument('--save_dir', type=str, default='./MVTEC2_results')
    parser.add_argument('--save_name', type=str,
                        default='mvtec')
    args = parser.parse_args()
    #
    item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
     'toothbrush', 'transistor', 'wood', 'zipper']
    # item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
    #             'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    # item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    # 'toothbrush', 'transistor', 'wood', 'zipper', 'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
    #             'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    print(torch.cuda.device_count())
    train(item_list)
