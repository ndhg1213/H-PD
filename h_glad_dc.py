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
    # 设置随机数种子使每次实验结果相同
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), 'GLaD-DC')

    # 设定蒸馏方法的保存路径
    args.save_path = os.path.join(args.save_path, "dc", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    # 设定评估epoch
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments

    # 为模型评估池中每个关键字添加列表
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    args.distributed = torch.cuda.device_count() > 1
    # 在像素级优化不需要使用GAN
    if args.space == 'p':
        G, zdim = None, None
    # latent的采样空间来自styleGAN_xl的w空间
    elif args.space == 'wp':
        G, zdim, w_dim, num_ws = load_sgxl(args.res, args)

    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)
    
    origin_features = get_feature(num_classes=num_classes, indices_class=indices_class, images_all=images_all, 
                                  channel=channel, im_size=im_size, DiffAugment=DiffAugment, args=args)

    real_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True,
                                                    num_workers=16)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)

    latents, f_latents, label_syn = prepare_latents(layer=1, channel=channel, num_classes=num_classes, im_size=im_size, zdim=zdim, G=G, class_map_inv=class_map_inv, get_images=get_images, args=args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    print('%s training begins' % get_time())

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    eval_pool_dict = get_eval_lrs(args)

    save_this_it = False
    
    num_layers = latents.shape[1] - 1
    best_layer = 0
    best_score = 0
    flag = 0
    # _, best_score = choose_optimal(layer=1, latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, 
    #                                best_score=best_score, testloader=testloader, model_eval_pool=model_eval_pool, 
    #                                channel=channel, num_classes=num_classes,
    #                                im_size=im_size, args=args)
    
    scores = []
    
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

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.

            for ol in range(args.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                if args.space == "wp":
                    with torch.no_grad():
                        image_syn_w_grad = torch.cat([latent_to_im(layer, G, (syn_image_split, f_latents_split), args) for
                                           syn_image_split, f_latents_split, label_syn_split in
                                           zip(torch.split(latents, args.sg_batch),
                                               torch.split(f_latents, args.sg_batch),
                                               torch.split(label_syn, args.sg_batch))])
                else:
                    image_syn_w_grad = latents

                # 根据论文说法先获得不含梯度计算图的图片S
                if args.space == "wp":
                    image_syn = image_syn_w_grad.detach()
                    image_syn.requires_grad_(True)
                else:
                    image_syn = image_syn_w_grad
                ''' update synthetic data '''

                optimizer_img.zero_grad()
                for c in range(num_classes):
                    loss = torch.tensor(0.0).to(args.device)
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    # 计算真实数据集上的梯度
                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    # 计算蒸馏集上的梯度
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    # 计算L_dc
                    loss = match_loss(gw_syn, gw_real, args)

                    # 这一步按照论文只反传L对S的梯度
                    loss.backward()
                    loss_avg += loss.item()
                    del img_real, output_real, loss_real, gw_real, output_syn, loss_syn, gw_syn, loss

                if args.space == "wp":
                    # this method works in-line and back-props gradients to latents and f_latents
                    gan_backward(layer=layer, latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

                else:
                    latents.grad = image_syn.grad.detach().clone()

                # 完成一轮对图片的训练
                optimizer_img.step()
                optimizer_img.zero_grad()

                if ol == args.outer_loop - 1:
                    break

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

                # 在内循环使用蒸馏集训练网络
                for il in range(args.inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

            # 计算一类图片的平均损失
            loss_avg /= (num_classes*args.outer_loop)
            
            # 计算蒸馏数据与原数据集的特征距离 
            dist = get_dist(num_classes=num_classes, real_features=origin_features, image_syn=image_syn, 
                            channel=channel, im_size=im_size, DiffAugment=DiffAugment, args=args)
            
            if dist < dist_min:
                dist_min = dist
                record = it

            if it % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                
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
            
        # print(save_this_it, best_layer, best_score)
        
        # scores.append(score)
                
        f_latents = update_latents(latents=latents, f_latents=f_latents, G=G, current_layer=layer, args=args)
    
    print('best layer {}, best score {} at iteration {}'.format(best_layer, best_score, flag))
    print(scores)



if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()

    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=0.001, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for gan weights')

    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--inner_loop', type=int, default=1, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    
    parser.add_argument('--inter_iteration', type=int, default=100, help='inter training iterations')
    
    args = parser.parse_args()

    main(args)

