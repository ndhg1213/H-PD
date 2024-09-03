import torch
import numpy as np
import copy
import utils
import os
import torchvision
import gc
from tqdm import tqdm

from utils import get_network, config, evaluate_synset


def build_dataset(ds, class_map, num_classes):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    # 使用tqdm添加一个进度条
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    return images_all, labels_all, indices_class


def prepare_latents(layer=None, channel=3, num_classes=10, im_size=(32, 32), zdim=512, G=None, class_map_inv={}, get_images=None, args=None):
    with torch.no_grad():
        ''' initialize the synthetic data '''
        # 生成num_classes * ipc 数量的顺序标签
        label_syn = torch.tensor([i*np.ones(args.ipc, dtype=np.int64) for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.space == 'p':
            # 优化空间为像素级即不使用prior直接随机初始化latent
            latents = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=False, device=args.device)
            f_latents = None

        else:
            # 随机初始化使用prior的优化空间为（类别数，ipc，styleGAN_xl的latent_dim）
            zs = torch.randn(num_classes * args.ipc, zdim, device=args.device, requires_grad=False)

            if "imagenet" in args.dataset:
                one_hot_dim = 1000
            elif args.dataset == "CIFAR10":
                one_hot_dim = 10
            elif args.dataset == "CIFAR100":
                one_hot_dim = 100
            # 需要进行平均操作
            if args.avg_w:
                G_labels = torch.zeros([label_syn.nelement(), one_hot_dim], device=args.device)
                # 将标签初始化得到一个（标签数量，标签数量）的矩阵
                G_labels[
                    torch.arange(0, label_syn.nelement(), dtype=torch.long), [class_map_inv[x.item()] for x in
                                                                         label_syn]] = 1
                new_latents = []
                for label in G_labels:
                    # 重写zs大小为（1000，512）
                    zs = torch.randn(1000, zdim).to(args.device)
                    # 将标签转换为（1000，数据集类别数）并通过GAN的mapping network进行转换为w
                    ws = G.mapping(zs, torch.stack([label] * 1000))
                    w = torch.mean(ws, dim=0)
                    new_latents.append(w)
                latents = torch.stack(new_latents)
                del zs
                for _ in new_latents:
                    del _
                del new_latents


            else:
                G_labels = torch.zeros([label_syn.nelement(), one_hot_dim], device=args.device)
                G_labels[
                    torch.arange(0, label_syn.nelement(), dtype=torch.long), [class_map_inv[x.item()] for x in
                                                                              label_syn]] = 1
                if args.distributed and False:
                    latents = G.mapping(zs.to("cuda:1"), G_labels.to("cuda:1")).to("cuda:0")
                else:
                    latents = G.mapping(zs, G_labels)
                del zs

            del G_labels

            # latent的来源是将标签作为不同类别图片的z输入styleGAN_xl的mapping_network中得到w
            ws = latents
            # layer表示Fn空间中的起始层数
            # 必然从第零层开始准备隐编码
            if True:
                # 将w送入styleGAN_xl的synthesis_network中得到f
                f_latents = torch.cat(
                    [G.forward(split_ws, f_layer=layer, mode="to_f").detach() for split_ws in
                     torch.split(ws, args.sg_batch)])
                f_type = f_latents.dtype
                f_latents = f_latents.to(torch.float32).cpu()
                # 转换nan
                f_latents = torch.nan_to_num(f_latents, posinf=5.0, neginf=-5.0)
                # 约束值范围
                f_latents = torch.clip(f_latents, min=-10, max=10)
                f_latents = f_latents.to(f_type).cuda()

                print(torch.mean(f_latents), torch.std(f_latents))

                if args.rand_f:
                    # 随机初始化f时添加图片标签信息转换得到f的均值与方差约束
                    f_latents = (torch.randn(f_latents.shape).to(args.device) * torch.std(
                        f_latents, dim=(1,2,3), keepdim=True) + torch.mean(f_latents, dim=(1,2,3), keepdim=True))

                f_latents = f_latents.to(f_type)
                print(torch.mean(f_latents), torch.std(f_latents))
                f_latents.requires_grad_(True)

        # 在真实图片中采样并且优化空间为像素级即原本数据蒸馏方法
        if args.pix_init == 'real' and args.space == "p":
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                latents.data[c*args.ipc:(c+1)*args.ipc] = torch.cat([get_images(c, 1).detach().data for s in range(args.ipc)])
        else:
            print('initialize synthetic data from random noise')

        latents = latents.detach().to(args.device).requires_grad_(True)

        # latents对应w+空间，f_latents对应f空间，label_syn对应蒸馏数据的标签
        return latents, f_latents, label_syn


def get_optimizer_img(latents=None, f_latents=None, G=None, args=None):
    if args.space == "wp":
        # 根据论文SGD优化对象为w
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_w, momentum=0.5)
        optimizer_img.add_param_group({'params': f_latents, 'lr': args.lr_img, 'momentum': 0.5})
    else:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_img, momentum=0.5)

    if args.learn_g:
        G.requires_grad_(True)
        optimizer_img.add_param_group({'params': G.parameters(), 'lr': args.lr_g, 'momentum': 0.5})

    optimizer_img.zero_grad()

    return optimizer_img

def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict

# 装载styleGAN_xl
def load_sgxl(res, args=None):
    import sys
    import os
    p = os.path.join("stylegan_xl")
    if p not in sys.path:
        sys.path.append(p)
    import dnnlib
    import legacy
    from sg_forward import StyleGAN_Wrapper
    device = torch.device('cuda')
    if args.special_gan is not None:
        if args.special_gan == "ffhq":
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq{}.pkl".format(res)
            network_pkl = "../stylegan_xl/ffhq{}.pkl".format(res)
            key = "G_ema"
        elif args.special_gan == "pokemon":
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon{}.pkl".format(res)
            network_pkl = "../stylegan_xl/pokemon{}.pkl".format(
                res)
            key = "G_ema"

    elif "imagenet" in args.dataset:
        if args.rand_gan_con:
            network_pkl = "../stylegan_xl/random_conditional_{}.pkl".format(res)
            key = "G"
        elif args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_{}.pkl".format(res)
            key = "G"
        else:
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet{}.pkl".format(res)
            network_pkl = "./stylegan_xl/imagenet{}.pkl".format(res)
            key = "G_ema"
    elif args.dataset == "CIFAR10":
        if args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_32.pkl"
            key = "G"
        else:
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl"
            network_pkl = "./stylegan_xl/cifar10.pkl"
            key = "G_ema"
    elif args.dataset == "CIFAR100":
        if args.rand_gan_con:
            network_pkl = "../stylegan_xl/random_conditional_32.pkl"
            key = "G"
        elif args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_32.pkl"
            key = "G"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)[key]
        G = G.eval().requires_grad_(False).to(device)

    z_dim = G.z_dim
    w_dim = G.w_dim
    num_ws = G.num_ws

    G.eval()
    mapping = G.mapping
    G = StyleGAN_Wrapper(G)
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:

        G = nn.DataParallel(G)
        mapping = nn.DataParallel(mapping)

    G.mapping = mapping

    return G, z_dim, w_dim, num_ws


# 使用GAN将latent转为图片
def latent_to_im(layer, G, latents, args=None):

    if args.space == "p":
        return latents

    mean, std = config.mean, config.std

    if "imagenet" in args.dataset:
        class_map = {i: x for i, x in enumerate(config.img_net_classes)}

        # 像素级空间优化图片就是隐编码
        if args.space == "p":
            im = latents

        elif args.space == "wp":
            # 从GAN的f空间生成图片
            im = G(latents[0], latents[1], layer, mode="from_f")

        im = (im + 1) / 2
        # im = (im - mean) / std

    elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        if args.space == "p":
            im = latents
        elif args.space == "wp":
            im = G(latents[0], latents[1], layer, mode="from_f")

            if args.distributed and False:
                mean, std = config.mean_1, config.std_1

        im = (im + 1) / 2
        im = (im - mean) / std

    return im


def image_logging(it=None, layer=None, latents=None, f_latents=None, label_syn=None, G=None, save_this_it=None, args=None):
    with torch.no_grad():
        image_syn = latents.cuda()

        if args.space == "wp":
            with torch.no_grad():
                image_syn = torch.cat(
                    [latent_to_im(layer, G, (image_syn_split.detach(), f_latents_split.detach()), args=args).detach() for
                     image_syn_split, f_latents_split, label_syn_split in
                     zip(torch.split(image_syn, args.sg_batch),
                         torch.split(f_latents, args.sg_batch),
                         torch.split(label_syn, args.sg_batch))])
        
        print(image_syn.shape)
        save_dir = os.path.join(args.logdir, args.dataset, 'MTT', "{0:05d}".format(layer))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(10):
            st = "images_{0:05d}".format(layer) + "_{0:04d}".format(it) + "_{0:01d}.png".format(i)
            torchvision.utils.save_image(image_syn[i].cpu(), os.path.join(save_dir, st))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(layer)))

        if save_this_it:
            torchvision.utils.save_image(image_syn.cpu(), os.path.join(save_dir, "images_best.png".format(layer)))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(layer)))

        if args.ipc < 50 or args.force_save:

            upsampled = image_syn
            if "imagenet" not in args.dataset:
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)

            for clip_val in []:
                upsampled = torch.clip(image_syn, min=-clip_val, max=clip_val)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)

            for clip_val in [2.5]:
                std = torch.std(image_syn)
                mean = torch.mean(image_syn)
                upsampled = torch.clip(image_syn, min=mean - clip_val * std, max=mean + clip_val * std)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)

    del upsampled, grid


def gan_backward(layer=None, latents=None, f_latents=None, image_syn=None, G=None, args=None):
    f_latents.grad = None
    latents_grad_list = []
    f_latents_grad_list = []
    for latents_split, f_latents_split, dLdx_split in zip(torch.split(latents, args.sg_batch),
                                                          torch.split(f_latents, args.sg_batch),
                                                          torch.split(image_syn.grad, args.sg_batch)):
        
        latents_detached = latents_split.detach().clone().requires_grad_(True)
        f_latents_detached = f_latents_split.detach().clone().requires_grad_(True)

        # 重新使用GAN生成图片以获得S对Z的梯度
        syn_images = latent_to_im(layer, G=G, latents=(latents_detached, f_latents_detached), args=args)

        # 通过链式法则反传S对Z的梯度
        syn_images.backward((dLdx_split,))

        latents_grad_list.append(latents_detached.grad)
        f_latents_grad_list.append(f_latents_detached.grad)

        del syn_images
        del latents_split
        del f_latents_split
        del dLdx_split
        del f_latents_detached
        del latents_detached

        gc.collect()

    latents.grad = torch.cat(latents_grad_list)
    del latents_grad_list
    f_latents.grad = torch.cat(f_latents_grad_list)
    del f_latents_grad_list
    

def choose_optimal(layer=None, latents=None, f_latents=None, label_syn=None, G=None, best_score=None, testloader=None, model_eval_pool=[], channel=3, num_classes=10, im_size=(32, 32), args=None):

    save_this_it = False
    
    eval_pool_dict = get_eval_lrs(args)

    sum = 0

    for model_eval in model_eval_pool:
        
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s' % (
        args.model, model_eval))
        
        accs_test = []
        accs_train = []
        # 评估num_eval次
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, width=args.width, depth=args.depth,
                                       dist=False).to(args.device)  # get a random model
            eval_lats = latents
            eval_labs = label_syn
            image_syn = latents
            # 深拷贝防止意外修改
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                eval_labs.detach())  # avoid any unaware modification

            if args.space == "wp":
                with torch.no_grad():
                    image_syn_eval = torch.cat(
                        [latent_to_im(layer, G, (image_syn_eval_split, f_latents_split), args=args).detach() for
                          image_syn_eval_split, f_latents_split, label_syn_split in
                          zip(torch.split(image_syn_eval, args.sg_batch), torch.split(f_latents, args.sg_batch),
                          torch.split(label_syn, args.sg_batch))])

            args.lr_net = eval_pool_dict[model_eval]
            # 将蒸馏图片与标签送入评估蒸馏集方法
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader,
                                                         args=args, aug=True)
            del _
            del net_eval
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        print(accs_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)

        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std = np.std(np.max(accs_test, axis=1))

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
        len(accs_test[:, -1]), model_eval, acc_test_mean, acc_test_std))
        
        sum += acc_test_mean
    
    sum = sum / len(model_eval_pool)
    
    # 当前准确率刷新了该评估模式下最好准确率时记录
    if sum > best_score:
        save_this_it = True
    
    return save_this_it, sum


def update_latents(latents=None, f_latents=None, G=None, current_layer=None, args=None):
    ws = latents
    f_latents = torch.cat(
            [G.forward(split_ws, split_f, f_layer=current_layer, mode="from_to").detach() for split_ws, split_f in 
             zip(torch.split(ws, args.sg_batch), 
                 torch.split(f_latents, args.sg_batch))])
    f_type = f_latents.dtype
    f_latents = f_latents.to(torch.float32).cpu()
    # 转换nan
    f_latents = torch.nan_to_num(f_latents, posinf=5.0, neginf=-5.0)
    # 约束值范围
    f_latents = torch.clip(f_latents, min=-10, max=10)
    f_latents = f_latents.to(f_type).cuda()
    
    return f_latents

def get_feature(num_classes=10, indices_class=None, images_all=None, channel=3, im_size=(128, 128), DiffAugment=None, args=None):
    import time
    
    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)
    
    eval_num = 5
    indices = 10 * images_all.shape[0] // num_classes // args.batch_real
    
    real_features = None   
    for eval in range(eval_num):
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device)
        net.eval()
                
        index_features = None
        for index in range(indices):
            class_features = []
            for c in range(num_classes):
                img_real = get_images(c, args.batch_real // 10)
        
                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
        
                _, _, feature = net.forward_cafe(img_real)
                
                features = []
                for i in range(len(feature) - 1):
                    features.append(feature[i].detach())    
                
                class_features.append(features)
                del img_real, feature, features

            if index == 0:
                index_features = class_features
            else:
                for i in range(num_classes):
                    for j in range(len(index_features[i]) - 1):
                        index_features[i][j] += class_features[i][j]
            del class_features
            
        for i in range(num_classes):
            for j in range(len(index_features[i]) - 1):
                index_features[i][j] /= indices   
        
        if eval == 0:
            real_features = index_features
        else:
            for i in range(num_classes):
                for j in range(len(index_features[i]) - 1):
                    real_features[i][j] += index_features[i][j]

        del net, index_features
        
    for i in range(num_classes):
        for j in range(len(real_features[i]) - 1):
            real_features[i][j] /= eval_num   
                
    return real_features

def get_dist(num_classes=10, real_features=None, image_syn=None, channel=3, im_size=(128, 128), DiffAugment=None, args=None):
    import time
    import torch.nn as nn
    
    def criterion_layer(real_feature, syn_feature):
        MSE_LOSS = nn.MSELoss(reduction='sum')
        real_feature = torch.mean(real_feature, dim=0)
        syn_feature = torch.mean(syn_feature, dim=0)
        return MSE_LOSS(real_feature, syn_feature).item()
    
    image_syn_cloned = copy.deepcopy(image_syn.detach())
    dist = 0 
    
    net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device)
    net.eval()
    
    for c in range(num_classes):
        img_syn_cloned = image_syn_cloned[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
        
        if args.dsa:
            seed = int(time.time() * 1000) % 100000
            img_syn_cloned = DiffAugment(img_syn_cloned, args.dsa_strategy, seed=seed, param=args.dsa_param)
        

        _, _, syn_features = net.forward_cafe(img_syn_cloned)
        
        for i in range(len(syn_features) - 1):
            dist += criterion_layer(real_features[c][i], syn_features[i])
    
        del img_syn_cloned, syn_features
    del net
        
    return dist
