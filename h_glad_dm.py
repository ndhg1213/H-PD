import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug
from tqdm import tqdm
import torchvision
import random
import gc

from h_glad_utils import *

def main(args):

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), 'GLaD-DM')

    args.save_path = os.path.join(args.save_path, "dm", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    args.distributed = torch.cuda.device_count() > 1

    if args.space == 'p':
        G, zdim = None, None
    elif args.space == 'wp':
        G, zdim, w_dim, num_ws = load_sgxl(args.res, args)

    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)
    
    # origin_features = get_feature(num_classes=num_classes, indices_class=indices_class, images_all=images_all, 
    #                               channel=channel, im_size=im_size, DiffAugment=DiffAugment, args=args)

    real_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True,
                                                    num_workers=16)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)

    # 拿到最初delatents和f_latents，经过layer0
    latents, f_latents, label_syn = prepare_latents(layer=1, channel=channel, num_classes=num_classes, im_size=im_size,
                                                    zdim=zdim, G=G, class_map_inv=class_map_inv, get_images=get_images,
                                                    args=args)    

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)



    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    print('%s training begins'%get_time())

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    save_this_it = False
    
    num_layers = latents.shape[1] - 1
    
    best_layer = 0
    best_score = 0
    # _, best_score = choose_optimal(layer=1, latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, 
    #                                best_score=best_score, testloader=testloader, 
    #                                model_eval_pool=model_eval_pool, channel=channel, num_classes=num_classes,
    #                                im_size=im_size, args=args)
    
    scores = []
            
    # 对每层进行搜索
    for layer in range(1, num_layers):
        # 记录每一层的latents和f_latents
        results = []
        dist_min = float("inf")
        record = 0
        print(latents.shape, f_latents.shape, layer)
        
        optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=G, args=args)
        
        # 蒸馏过程
        for it in range(args.inter_iteration + 1):
            
            results.append((latents, f_latents))

            if it == 0:
                image_logging(it=0, layer=layer, latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, save_this_it=save_this_it, args=args)

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
            net.train()
            # 对于dm方法不需要模型参数的梯度
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

            loss_avg = 0

            if args.space == "wp":
                with torch.no_grad():
                    image_syn_w_grad = torch.cat([latent_to_im(layer, G, (syn_image_split, f_latents_split), args) for
                                                  syn_image_split, f_latents_split, label_syn_split in
                                                  zip(torch.split(latents, args.sg_batch),
                                                      torch.split(f_latents, args.sg_batch),
                                                      torch.split(label_syn, args.sg_batch))])
            else:
                image_syn_w_grad = latents

            if args.space == "wp":
                # 舍弃计算图
                image_syn = image_syn_w_grad.detach()
                image_syn.requires_grad_(True)
            else:
                image_syn = image_syn_w_grad
                image_syn.requires_grad_(True)

            ''' update synthetic data '''
            if 'BN' not in args.model: # for ConvNet
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real).to(args.device)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    # 使用ConvNet的embed提取原始数据与蒸馏数据的特征分布
                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            # 对于含有BN层的ConvNet需要送入所有的真实数据
            else: # for ConvNetBN
                images_real_all = []
                images_syn_all = []
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    images_real_all.append(img_real)
                    images_syn_all.append(img_syn)

                images_real_all = torch.cat(images_real_all, dim=0)
                images_syn_all = torch.cat(images_syn_all, dim=0)

                output_real = embed(images_real_all).detach()
                output_syn = embed(images_syn_all)

                # 计算dm方法的loss
                loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

            optimizer_img.zero_grad()
            # 反传L对S的梯度
            loss.backward()

            if args.space == "wp":
                # this method works in-line and back-props gradients to latents and f_latents
                gan_backward(layer=layer, latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

            else:
                latents.grad = image_syn.grad.detach().clone()

            optimizer_img.step()
            loss_avg += loss.item()


            loss_avg /= (num_classes)
            
            # dist = get_dist(num_classes=num_classes, real_features=origin_features, image_syn=image_syn, 
            #                 channel=channel, im_size=im_size, DiffAugment=DiffAugment, args=args)
            
            # if dist < dist_min:
            #     dist_min = dist
            #     record = it
                
            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])

        # 此时的隐编码是该层中distance最小的
        if args.search:
            latents, f_latents = results[record]
        
        # 使用评估模型选出最好的层
        # save_this_it, score = choose_optimal(layer=layer, latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, 
        #                                           best_score=best_score, testloader=testloader, model_eval_pool=model_eval_pool, 
        #                                           channel=channel, num_classes=num_classes,
        #                                           im_size=im_size, args=args)
        
        # if save_this_it:
        #     best_layer = layer 
        #     best_score = score
            
        # scores.append(score)
            
        # print(save_this_it, best_layer, best_score)
        
        f_latents = update_latents(latents=latents, f_latents=f_latents, G=G, current_layer=layer, args=args)
    
    print('best layer {}, best score {}'.format(best_layer, best_score))
    print(scores)


if __name__ == '__main__':
    if __name__ == '__main__':
        import shared_args

        parser = shared_args.add_shared_args()

        parser.add_argument('--lr_img', type=float, default=10, help='learning rate for pixels or f_latents')
        parser.add_argument('--lr_w', type=float, default=0.01, help='learning rate for updating synthetic latent w')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for gan weights')
        
        parser.add_argument('--inter_iteration', type=int, default=10, help='inter training iterations')
        
        args = parser.parse_args()

        main(args)


