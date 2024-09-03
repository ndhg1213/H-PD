import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset,  match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug

# import wandb
import copy
import random
from reparam_module import ReparamModule
import torch.utils.data
import warnings
import gc

from h_glad_utils import *

import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    args.lr_net = args.lr_teacher

    args.num_gpus = torch.cuda.device_count()

    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.lr_img is None:
        if args.space == 'p':
            args.lr_img = 0.1
        else:
            args.lr_img = 0.01

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.res, args=args)
    
    # images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)
    
    # origin_features = get_feature(num_classes=num_classes, indices_class=indices_class, images_all=images_all, 
    #                               channel=channel, im_size=im_size, DiffAugment=DiffAugment, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), 'GLaD-MTT')

    args.save_path = os.path.join(args.save_path, "mtt", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    args.dsa_param = dsa_params

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    if args.space == 'p':
        G, zdim = None, None

    elif args.space == 'wp':
        G, zdim, w_dim, num_ws = load_sgxl(args.res, args)

    if args.space == "p" and args.pix_init == "real":
        images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)

        real_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True,
                                                      num_workers=16)

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)


    latents, f_latents, label_syn = prepare_latents(layer=1, channel=channel, num_classes=num_classes, im_size=im_size, zdim=zdim, G=G, class_map_inv=class_map_inv, get_images=get_images, args=args)
    
    syn_lr = torch.tensor(args.lr_teacher, requires_grad=True).to(args.device)

    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    # 创建专家轨迹的路径
    expert_dir = os.path.join(args.buffer_path, args.dataset)

    expert_dir = os.path.join(expert_dir, args.model)

    expert_dir = os.path.join(expert_dir, "depth-{}".format(args.depth), "width-{}".format(args.width))
    
    expert_dir = os.path.join(expert_dir, "instancenorm")

    print("Expert Dir: {}".format(expert_dir))
    if args.load_all:
        # 加载全部专家轨迹
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1

    else:
        # 加载部分专家轨迹
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        
        # buffer随机加载pt文件
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
            
        # 此时buffer包含一个pt文件中的多条轨迹
        random.shuffle(buffer)

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}
    
    save_this_it = False
    
    num_layers = latents.shape[1] - 1
    
    best_layer = 0
    best_score = 0
    scores = []
    # _, best_score = choose_optimal(layer=1, latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, 
    #                                best_score=best_score, testloader=testloader, model_eval_pool=model_eval_pool, 
    #                                channel=channel, num_classes=num_classes,
    #                                im_size=im_size, args=args)

    for layer in range(1, num_layers):
        # 记录每一层的latents和f_latents
        results = []
        dist_min = float("inf")
        record = 0
        print(latents.shape, f_latents.shape, layer)
        
        optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=G, args=args)
    
        for it in range(args.inter_iteration + 1):
            
            results.append((latents, f_latents))

            if it == 0:
                image_logging(it=0, layer=layer, latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, save_this_it=save_this_it, args=args)

            student_net = get_network(args.model, channel, num_classes, im_size, width=args.width, depth=args.depth, dist=False).to(args.device)  # get a random model

            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

            if args.load_all:
                expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            else:
                # expert_trajectory包含一条专家轨迹
                expert_trajectory = buffer[expert_idx]
                expert_idx += 1
                # 一个pt文件中的所有轨迹加载完毕
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    # 当buffer_mtt中Iteration小于ditill_mtt中Iteration时需要重新加载轨迹
                    if file_idx == len(expert_files):
                        file_idx = 0
                        # 当所有的pt文件都加载结束后重新打乱
                        random.shuffle(expert_files)
                    # print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1 and len(expert_files) > 1:
                        del buffer
                        buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        buffer = buffer[:args.max_experts]
                    random.shuffle(buffer)

            # 专家轨迹t的取值为0~5
            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]

            # 获得专家轨迹的t+M时参数
            target_params = expert_trajectory[start_epoch+args.expert_epochs]
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

            # 获得学生轨迹的t时参数
            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            # 获得专家轨迹的t时参数
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

            student_net = ReparamModule(student_net)

            gradient_sum = torch.zeros(starting_params.shape).requires_grad_(False).to(args.device)

            param_dist = torch.tensor(0.0).to(args.device)

            # 获得轨迹匹配loss的分母
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            if args.distributed:
                student_net = torch.nn.DataParallel(student_net)

            student_net.train()

            syn_images = latents[:]
            if args.space == "wp":
                with torch.no_grad():
                    syn_images = torch.cat([latent_to_im(layer, G, (syn_image_split.detach(), f_latents_split.detach()),
                                                        args=args).detach() for
                                           syn_image_split, f_latents_split, label_syn_split in
                                           zip(torch.split(syn_images, args.sg_batch),
                                               torch.split(f_latents, args.sg_batch),
                                               torch.split(label_syn, args.sg_batch))])
                syn_images.requires_grad_(True)

            image_syn = syn_images.detach()

            y_hat = label_syn
            x_list = []
            y_list = []
            indices_chunks = []
            indices_chunks_copy = []
            original_x_list = []
            gc.collect()

            syn_label_grad = torch.zeros(label_syn.shape).to(args.device).requires_grad_(False)
            syn_images_grad = torch.zeros(syn_images.shape).requires_grad_(False).to(args.device)

            for il in range(args.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()
                indices_chunks_copy.append(these_indices.clone())

                x = syn_images[these_indices]
                this_y = y_hat[these_indices]

                original_x_list.append(x)

                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

                x_list.append(x.clone())
                y_list.append(this_y.clone())

                forward_params = student_params[-1]

                forward_params = copy.deepcopy(forward_params.detach()).requires_grad_(True)

                if args.distributed:
                    forward_params_expanded = forward_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params_expanded = forward_params

                x = student_net(x, flat_param=forward_params_expanded)

                ce_loss = criterion(x, this_y)

                # 根据交叉熵loss计算梯度更新学生网络参数获得学生训练轨迹
                grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]
                student_params.append(forward_params - syn_lr.item() * grad.detach().clone())
                gradient_sum = gradient_sum + grad.detach().clone()


            for il in range(args.syn_steps):
                w = student_params[il]

                if args.distributed:
                    w_expanded = w.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    w_expanded = w

                output = student_net(x_list[il], flat_param=w_expanded)

                if args.batch_syn:
                    ce_loss = criterion(output, y_list[il])
                else:
                    ce_loss = criterion(output, y_hat)

                grad = torch.autograd.grad(ce_loss, w, create_graph=True, retain_graph=True)[0]

                # Square term gradients.
                square_term = syn_lr.item() ** 2 * (grad @ grad)
                single_term = 2 * syn_lr.item() * grad @ (
                            syn_lr.item() * (gradient_sum - grad.detach().clone()) - starting_params + target_params)

                per_batch_loss = (square_term + single_term) / param_dist
                gradients = torch.autograd.grad(per_batch_loss, original_x_list[il], retain_graph=False)[0]

                # 根据内循环训练轨迹计算生成图片对于loss的梯度
                with torch.no_grad():
                    syn_images_grad[indices_chunks_copy[il]] += gradients

            # ---------end of computing input image gradients and learning rates--------------

            del w, output, ce_loss, grad, square_term, single_term, per_batch_loss, gradients, student_net, w_expanded, forward_params, forward_params_expanded

            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()

            syn_lr.requires_grad_(True)
            # 计算学生轨迹起始参数 - 下降梯度 - 专家目标参数
            grand_loss = starting_params - syn_lr * gradient_sum - target_params
            grand_loss = grand_loss.dot(grand_loss)
            grand_loss = grand_loss / param_dist

            lr_grad = torch.autograd.grad(grand_loss, syn_lr)[0]
            syn_lr.grad = lr_grad

            optimizer_lr.step()
            optimizer_lr.zero_grad()

            image_syn.requires_grad_(True)

            image_syn.grad = syn_images_grad.detach().clone()

            del syn_images_grad
            del lr_grad

            for _ in student_params:
                del _
            for _ in x_list:
                del _
            for _ in y_list:
                del _


            torch.cuda.empty_cache()

            gc.collect()

            if args.space == "wp":
                # this method works in-line and back-props gradients to latents and f_latents
                gan_backward(layer=layer, latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

            else:
                latents.grad = image_syn.grad.detach().clone()

            optimizer_img.step()
            optimizer_img.zero_grad()

            # dist = get_dist(num_classes=num_classes, real_features=origin_features, image_syn=image_syn, 
            #                 channel=channel, im_size=im_size, DiffAugment=DiffAugment, args=args)
            
            # if dist < dist_min:
            #     dist_min = dist
            #     record = it

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
    
        # 此时的隐编码是该层中loss最小的
        if args.search:
            latents, f_latents = results[record]        
        
        # 评估本层的latents和f_latents
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
    # wandb.finish()


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true')
    parser.add_argument('--max_start_epoch', type=int, default=5)
    parser.add_argument('--max_files', type=int, default=None)
    parser.add_argument('--max_experts', type=int, default=None)
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')

    parser.add_argument('--lr_img', type=float, default=10000 , help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=10, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_lr', type=float, default=1e-06, help='learning rate learning rate')
    parser.add_argument('--lr_g', type=float, default=0.1, help='learning rate for gan weights')
    
    parser.add_argument('--inter_iteration', type=int, default=100, help='inter training iterations')

    args = parser.parse_args()

    main(args)

