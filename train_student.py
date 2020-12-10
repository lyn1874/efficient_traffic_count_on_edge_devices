import datetime
import os
import argparse
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights

path_mom = "/home/jovyan/bo/"

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
    

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=str2bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=str, default="multistep")
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--subtract_bg', type=str2bool, default=False)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=1141, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--datadir', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--saved_path', type=str, default=path_mom + '/exp_data/mot16Det/')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of trainging, '
                                                                  'the output images will be in test/')
    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(opt):
    params = Params(f'parameter/{opt.project}.yml')
    print(params)
    print("The number of classes", len(params.obj_list))
    use_background = params.use_background
    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    print("cuda is available or not-------------", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    
    opt.saved_path = opt.saved_path + \
        "/D%d/%s_%s_lr_%.4f_decay_%s_optim_%s_subtractBG_%s_head_%s_version_%d" % (opt.compound, 
                                                                                   params.project_name, 
                                                                                   params.train_set, opt.lr, opt.lr_decay,
                                                                                   opt.optim, opt.subtract_bg, opt.head_only, 
                                                                                   opt.version) 
    if opt.load_weights:
        opt.saved_path = opt.saved_path + '/'
    else:
        opt.saved_path = opt.saved_path + '_from_scratch/'
    opt.log_path = opt.saved_path + "tensorboard/"
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers,
                       'pin_memory': True}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers,
                  'pin_memory': True}
    
    background = [0.0, 0.0, 0.0]
    val_background = [0.0, 0.0, 0.0]
    mean = params.mean
    std = params.std
    if opt.subtract_bg == True:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    print("The training path", opt.datadir + params.project_name, params.train_set)
    print("The validation path", opt.datadir + params.project_name, params.val_set)
    print("normalization mean.......", mean)
    print("normalization std........", std)

    training_set_full = CocoDataset(root_dir=opt.datadir + params.project_name + '/', set=params.train_set,
                                    subtract_bg=opt.subtract_bg, 
                                    transform=transforms.Compose([Normalizer(mean=mean, std=std, 
                                                                  background_im=background),
                                                                  Augmenter(),
                                                                  Resizer(input_sizes[opt.compound])]))
    training_generator = DataLoader(training_set_full, **training_params)

    if params.val_set != "none":
        val_set = CocoDataset(root_dir=opt.datadir + params.project_name + '/', set=params.val_set,
                              subtract_bg=opt.subtract_bg, 
                              transform=transforms.Compose([Normalizer(mean=mean, std=std,
                                                                       background_im=val_background),
                                                            Resizer(input_sizes[opt.compound])]))
        val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        print("----I am converting my model to gpu---")
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    if opt.lr_decay == "multistep":
        if opt.compound == 3:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 14])  #10, 25
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30])  #10, 25
    elif opt.lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
    elif opt.lr_decay == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    else:
        print("----LR decay method %s does not exist--------" % opt.lr_decay)
    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    print("There are %d frames with objects" % (len(training_set_full)))
    print("There are supposed to be %d iterations in per epoch" % ((len(training_set_full)) // opt.batch_size))
    print("There are actually %d iterations per epoch" % num_iter_per_epoch)
    opt.save_interval = num_iter_per_epoch

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                
                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))
                    
                    writer.add_scalar('Reg', reg_loss, step)
                    writer.add_scalar('Cls', cls_loss, step)
                    writer.add_scalar('Tot', loss, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and epoch % 4 == 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            if opt.lr_decay == "plateau":
                scheduler.step(np.mean(epoch_loss))  # this is used when it's reducelronPlateau ?
            else:
                scheduler.step()

            if epoch % opt.val_interval == 0 and params.val_set != "none":
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']
                    
                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                writer.add_scalar('val_tot', loss, step)
                writer.add_scalar('val_reg', reg_loss, step)
                writer.add_scalar('val_cls', cls_loss, step)

                if loss + opt.es_min_delta < best_loss and epoch % 4 == 0:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound}_{epoch}_{step}.pth')

                    # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, loss))
                    break
            if epoch == 0:
                writer.add_images('Input_image', imgs.abs(), epoch)
                print(imgs.min(), imgs.max())

    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    if opt.compound <= 2:
        opt.lr = 5e-3
        opt.num_epochs = 50
    else:
        opt.lr = 1e-3
        opt.num_epochs = 20
    if opt.compound == 0:
        opt.batch_size = 24
    elif opt.compound == 1:
        if opt.head_only == True:
            opt.batch_size = 24
        else:
            opt.batch_size = 20
    elif opt.compound == 2:
        if opt.head_only == True:
            opt.batch_size = 20
        else:
            opt.batch_size = 14
    else:
        opt.batch_size = [4 if opt.head_only == False else 16][0]
    print("-------------------------------------------------------------------")
    print("------------------argument for current experiment------------------")
    print("-------------------------------------------------------------------")
    for arg in vars(opt):
        print(arg, getattr(opt, arg))
    print("-------------------------------------------------------------------")

    
    train(opt)

    