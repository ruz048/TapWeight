import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from datetime import datetime
from tqdm import tqdm
from dataloader.JigsawLoader import JigsawDataset, load_pretraining_dataset
from model.feat2image_model import generator, netlocalD
from model.model import ImageMol, Matcher
from collections import Counter
from dataloader.image_dataloader import ImageDataset, load_filenames_and_labels_multitask, get_datasets
from model.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from model.train_utils import fix_train_random_seed, load_smiles
from utils.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from utils.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from copy import deepcopy

from itertools import cycle

def load_norm_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_tra = [transforms.CenterCrop(args.imageSize),
               transforms.RandomHorizontalFlip(),
               transforms.RandomGrayscale(p=0.2),
               transforms.RandomRotation(degrees=360)]
    tile_tra = [transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor()]
    return normalize, img_tra, tile_tra

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol Reweighting')

    # basic
    parser.add_argument('--dataset', type=str, default="BBBP", help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./datasets/pretraining/", help='data root')

    parser.add_argument('--dataset_pt', type=str, default="BBBP", help='dataset name, e.g. BBBP, tox21, ...')
    parser.add_argument('--dataroot_pt', type=str, default="./datasets/pretraining/", help='data root')

    parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use (default: 1)')

    # optimizer
    parser.add_argument('--finetune_lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=-5, type=float, help='weight decay pow (default: -5)')

    # train

    parser.add_argument('--runseed', type=int, default=2021, help='random seed to run model (default: 2021)')
    parser.add_argument('--split', default="random", type=str,
                        choices=['random', 'stratified', 'scaffold', 'random_scaffold', 'scaffold_balanced'],
                        help='regularization of classification loss')

    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to checkpoint (default: None)')

    parser.add_argument('--image_model', type=str, default="ResNet18", help='e.g. ResNet18, ResNet34')
    parser.add_argument('--image_aug', action='store_true', default=False, help='whether to use data augmentation')
    parser.add_argument('--weighted_CE', action='store_true', default=False, help='whether to use global imbalanced weight')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='task type')

    # log
    parser.add_argument('--log_dir', default='./logs/finetune/', help='path to log')

    #pretrain
    parser.add_argument('--pretrain_lr', default=0.01, type=float, help='learning rate (default: 0.01)')
    
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')
    parser.add_argument('--val_workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', type=int, default=151, help='number of total epochs to run (default: 151)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch_pt', default=512, type=int, help='pretrain mini-batch size')
    parser.add_argument('--batch_ft', default=64, type=int, help='finetune mini-batch size')
    parser.add_argument('--batch_rt', default=64, type=int, help='reweight mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--checkpoints', type=int, default=1,
                        help='how many iterations between two checkpoints (default: 1)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    

    parser.add_argument('--ckpt_dir', default='./ckpts/pretrain_model', help='path to checkpoint')
    parser.add_argument('--modelname', type=str, default="ResNet18", choices=["ResNet18"], help='supported model')
    parser.add_argument('--verbose', action='store_true', help='')

    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to network')
    parser.add_argument('--Jigsaw_lambda', type=float, default=1,
                        help='start JPP task, 1 means start, 0 means not start')
    parser.add_argument('--cluster_lambda', type=float, default=1, help='start M3GC task')
    parser.add_argument('--constractive_lambda', type=float, default=0, help='start MCL task')
    parser.add_argument('--matcher_lambda', type=float, default=0, help='start MRD task')
    parser.add_argument('--is_recover_training', type=int, default=1, help='start MIR task')
    parser.add_argument('--cl_mask_type', type=str, default="rectangle_mask", help='',
                        choices=["random_mask", "rectangle_mask", "mix_mask"])
    parser.add_argument('--cl_mask_shape_h', type=int, default=16, help='mask_utils->create_rectangle_mask()')
    parser.add_argument('--cl_mask_shape_w', type=int, default=16, help='mask_utils->create_rectangle_mask()')
    parser.add_argument('--cl_mask_ratio', type=float, default=0.001, help='mask_utils->create_random_mask()')

    parser.add_argument('--num_losses', type=int, default=5, help='number of pretraining losses')
    parser.add_argument('--load_save', action='store_true', help='')
    parser.add_argument('--wandb', action='store_true', help='')
    parser.add_argument('--save_dir', default='/data1/ruiyi/molreweight/ckpt', help='path to checkpoint')
    parser.add_argument('--reweight_lr', default=0.1, type=float, help='reweigting learning rate (default: 0.01)')
    parser.add_argument('--val_freq', type=int, default=200)
    parser.add_argument('--iters', type=int, default=10000)
    parser.add_argument('--lam', default=1e-2, type=float, help='lambda')
    parser.add_argument('--same_ft_dataset', action='store_true', help='')

    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--unroll_steps", type=int, default=1)


    return parser.parse_args()

args = parse_args()
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.wandb:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="imagemol-task-reweight",
        config={
            "learning_rate_reweight": args.reweight_lr,
            "learning_rate_finetune": args.pretrain_lr,
            "learning_rate_pretrain": args.finetune_lr,
            'dataset_ft': args.dataset
        },
    )

print(args)

args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
args.verbose = True

device, device_ids = setup_device(args.ngpu)

# fix random seeds
fix_train_random_seed(args.runseed)

# architecture name
if args.verbose:
    print('Architecture: {}'.format(args.image_model))

##################################### initialize some parameters #####################################
if args.task_type == "classification":
    eval_metric = "rocauc"
    valid_select = "max"
    min_value = -np.inf
elif args.task_type == "regression":
    if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
        eval_metric = "mae"
    else:
        eval_metric = "rmse"
    valid_select = "min"
    min_value = np.inf
else:
    raise Exception("{} is not supported".format(args.task_type))

task_type=args.task_type

print("eval_metric: {}".format(eval_metric))
if args.image_aug:
    img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(p=0.2), transforms.RandomRotation(degrees=360),
                                 transforms.ToTensor()]
else:
    img_transformer_train = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
names, labels = load_filenames_and_labels_multitask(args.image_folder, args.txt_file, task_type=args.task_type)
names, labels = np.array(names), np.array(labels)
num_tasks = labels.shape[1]

if args.split == "random":
    train_idx, val_idx, test_idx = split_train_val_test_idx(list(range(0, len(names))), frac_train=0.8,
                                                            frac_valid=0.1, frac_test=0.1, seed=args.seed)
elif args.split == "stratified":
    train_idx, val_idx, test_idx = split_train_val_test_idx_stratified(list(range(0, len(names))), labels,
                                                                        frac_train=0.8, frac_valid=0.1,
                                                                        frac_test=0.1, seed=args.seed)
elif args.split == "scaffold":
    smiles = load_smiles(args.txt_file)
    train_idx, val_idx, test_idx = scaffold_split_train_val_test(list(range(0, len(names))), smiles, frac_train=0.8,
                                                                    frac_valid=0.1, frac_test=0.1)
elif args.split == "random_scaffold":
    smiles = load_smiles(args.txt_file)
    train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(list(range(0, len(names))), smiles,
                                                                        frac_train=0.8, frac_valid=0.1,
                                                                        frac_test=0.1, seed=args.seed)
elif args.split == "scaffold_balanced":
    smiles = load_smiles(args.txt_file)
    train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(list(range(0, len(names))), smiles,
                                                                            frac_train=0.8, frac_valid=0.1,
                                                                            frac_test=0.1, seed=args.seed,
                                                                            balanced=True)

if args.same_ft_dataset:
    train_val_idx=np.concatenate((train_idx,val_idx))
    train_idx = train_val_idx
    val_idx = train_val_idx

name_train, name_val, name_test, labels_train, labels_val, labels_test = names[train_idx], names[val_idx], names[
    test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset = ImageDataset(name_train, labels_train, img_transformer=transforms.Compose(img_transformer_train),
                                normalize=normalize, args=args)
val_dataset = ImageDataset(name_val, labels_val, img_transformer=transforms.Compose(img_transformer_test),
                            normalize=normalize, args=args)
test_dataset = ImageDataset(name_test, labels_test, img_transformer=transforms.Compose(img_transformer_test),
                            normalize=normalize, args=args)

finetune_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_ft,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
eval_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_rt,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.batch_rt,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True)

# default params
jigsaw_classes = 100 + 1
label1_classes = 100
label2_classes = 1000
label3_classes = 10000
val_size = 0.05
original_image_rate = 0.8
eval_each_batch = 1000

# load model
model = ImageMol(args.modelname, jigsaw_classes, label1_classes=label1_classes, label2_classes=label2_classes,
                    label3_classes=label3_classes)

cudnn.benchmark = True


# define loss function
criterion = torch.nn.CrossEntropyLoss().cuda()

criterionBCE = torch.nn.BCELoss().cuda()

# load data
normalize, img_tra, tile_tra = load_norm_transform()

name_train, name_val, labels_train, labels_val = load_pretraining_dataset(args.dataroot_pt, args.dataset_pt, val_size, args)
pretrain_dataset = JigsawDataset(name_train, labels_train, img_transformer=transforms.Compose(img_tra),
                                tile_transformer=transforms.Compose(tile_tra),
                                bias_whole_image=original_image_rate,
                                normalize=normalize,
                                args=args)

pretrain_dataloader = torch.utils.data.DataLoader(pretrain_dataset,
                                                batch_size=args.batch_pt,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)

# Custom IterableDataset to combine two dataloaders
class CombinedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataloader1, dataloader2):
        self.dataloader1 = dataloader1
        self.dataloader2 = dataloader2
    
    def __iter__(self):
        # Create the iterator for the dataloaders
        dataloader1_iter = iter(self.dataloader1)
        dataloader2_iter = cycle(self.dataloader2)
        
        try:
            while True:
                # Fetch the next batch from both dataloaders
                batch1 = next(dataloader1_iter)
                batch2 = next(dataloader2_iter)
                yield batch1, batch2
        except StopIteration:
            pass  # When dataloader1 is exhausted, stop iteration


combined_pretrain_dataloader = (pretrain_dataloader, finetune_dataloader)
weights = None
if args.task_type == "classification":
    if args.weighted_CE:
        labels_train_list = labels_train[labels_train != -1].flatten().tolist()
        count_labels_train = Counter(labels_train_list)
        imbalance_weight = {key: 1 - count_labels_train[key] / len(labels_train_list) for key in count_labels_train.keys()}
        weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
    criterion_ft = nn.BCEWithLogitsLoss(reduction="none")
elif args.task_type == "regression":
    criterion_ft = nn.MSELoss()
else:
    raise Exception("param {} is not supported.".format(args.task_type))

model_ft = load_model(args.image_model, imageSize=args.imageSize, num_classes=num_tasks)



if args.resume:
    if os.path.isfile(args.resume):  # only support ResNet18 when loading resume
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        
        ckp_keys = list(checkpoint['state_dict'])
        cur_keys = list(model_ft.state_dict())
        model_sd = model_ft.state_dict()
        if args.image_model == "ResNet18":
            ckp_keys = ckp_keys[:120]
            cur_keys = cur_keys[:120]

        for ckp_key, cur_key in zip(ckp_keys, cur_keys):
            model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

        model_ft.load_state_dict(model_sd)
        arch = checkpoint['arch']
        print("resume model info: arch: {}".format(arch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


model = model.cuda()


class Reweight(torch.nn.Module):
    def __init__(self):
        super(Reweight, self).__init__()
        self.weight=torch.nn.Parameter(torch.ones(args.num_losses))

    def forward(self):
        return torch.softmax(self.weight,0)

class Pretrain(torch.nn.Module):
    def __init__(self):
        super(Pretrain, self).__init__()
        self.imagemol=model
        self.ftfc = model_ft.fc
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, batch, is_pt):
        if is_pt:
            return self.imagemol(batch)
        else:

            x = self.imagemol.embedding_layer(batch)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.ftfc(x)

model_pretrain = Pretrain()

model_reweight = Reweight().to(device)


class Pretraining(ImplicitProblem):
    def training_step(self, batch):
        
        batch_pt, batch_ft = batch[0], batch[1]

        batch = batch_pt

        model=self.module

        Jigsaw_img, Jigsaw_label, original_label, data_non_mask, data64_non_mask, cl_data_mask, _ = batch
        Jigsaw_img_var = torch.autograd.Variable(Jigsaw_img.cuda())

        Jigsaw_label_var = torch.autograd.Variable(Jigsaw_label.cuda())

        data_non_mask = torch.autograd.Variable(data_non_mask.cuda())
        data64_non_mask = torch.autograd.Variable(data64_non_mask.cuda())
        cl_data_mask = torch.autograd.Variable(cl_data_mask.cuda())

        original_label1_var = torch.autograd.Variable(original_label[0].cuda())
        original_label2_var = torch.autograd.Variable(original_label[1].cuda())
        original_label3_var = torch.autograd.Variable(original_label[2].cuda())

        hidden_feat, pre_Jigsaw_label, pre_class_label1, pre_class_label2, pre_class_label3 = model(
            Jigsaw_img_var, True)
        Jig_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        if args.Jigsaw_lambda != 0:
            Jig_loss = criterion(pre_Jigsaw_label, Jigsaw_label_var)

        class_loss1 = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        class_loss2 = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        class_loss3 = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        class_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        if args.cluster_lambda != 0:
            class_loss1 = criterion(pre_class_label1, original_label1_var)
            class_loss2 = criterion(pre_class_label2, original_label2_var)
            class_loss3 = criterion(pre_class_label3, original_label3_var)
            class_loss = class_loss1 * self.reweight.module()[0]+ class_loss2 * self.reweight.module()[1]+ class_loss3* self.reweight.module()[2]

        hidden_feat_non_mask, _, _, _, _ = model(data_non_mask, True)
        hidden_feat_mask, _, _, _, _ = model(cl_data_mask, True)
        constractive_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        if args.constractive_lambda != 0:
            constractive_loss = (hidden_feat_non_mask - hidden_feat_mask).pow(2).sum(axis=1).sqrt().mean()
            AvgConstractiveLoss += constractive_loss.item() / len(pretrain_dataloader)

        reasonability_loss = torch.autograd.Variable(torch.Tensor([0.0])).cuda()

        errG = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        errD = torch.autograd.Variable(torch.Tensor([0.0])).cuda()
        
        
        # calculating all loss to backward
        loss = class_loss  + self.reweight.module()[3] * Jig_loss + self.reweight.module()[4] * constractive_loss + self.training_step_ft(batch_ft)
        #loss_ft = 
        if args.wandb:
            wandb.log({"pretraining loss": loss.item()})
            wandb.log({"pretraining lr": self.optimizer.param_groups[0]['lr']})
        else:print(loss.item())
        return loss
    def training_step_ft(self, batch):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        model=self.module
        pred = model(images, False)

        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion_ft(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            if weights is None:
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                cls_weights = labels.clone()
                cls_weights_mask = []
                for i, weight in enumerate(weights):
                    cls_weights_mask.append(cls_weights == i)
                for i, cls_weight_mask in enumerate(cls_weights_mask):
                    cls_weights[cls_weight_mask] = weights[i]
                loss = torch.sum(loss_mat * cls_weights) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion_ft(pred.double(), labels)
        return loss
    
    def configure_scheduler(self):
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.step_size, gamma=args.gamma
        )


class Reweighting(ImplicitProblem):
    def training_step(self, batch):
        model = self.pretrain.module
        images, labels = batch
        images, labels = images.to(device), labels.to(device)


        pred = model(images, False)
        labels = labels.view(pred.shape).to(torch.float64)
        if task_type == "classification":
            is_valid = labels != -1
            loss_mat = criterion_ft(pred.double(), labels)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            if weights is None:
                loss = torch.sum(loss_mat) / torch.sum(is_valid)
            else:
                cls_weights = labels.clone()
                cls_weights_mask = []
                for i, weight in enumerate(weights):
                    cls_weights_mask.append(cls_weights == i)
                for i, cls_weight_mask in enumerate(cls_weights_mask):
                    cls_weights[cls_weight_mask] = weights[i]
                loss = torch.sum(loss_mat * cls_weights) / torch.sum(is_valid)
        elif task_type == "regression":
            loss = criterion_ft(pred.double(), labels)

        if args.wandb:
            wandb.log({"reweighting loss": loss.item()})
            wandb.log({"reweighting lr": self.optimizer.param_groups[0]['lr']})
        else:print(loss.item())
        return loss

    def reg_loss(self):
        loss = 0

        for (n1, p1), (n2, p2) in zip(
            self.finetune.module.named_parameters(), self.pretrain.module.embedding_layer.named_parameters()
        ):
            loss = loss + args.lam * (p1 - p2).pow(2).sum()

        return loss
    def configure_scheduler(self):
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.step_size, gamma=args.gamma
        )


folder_save = '{}/{}'.format(args.save_dir,args.dataset)

# Check if the folder exists
if not os.path.exists(folder_save):
    # If the folder does not exist, create it
    os.makedirs(folder_save)

class LBIEngine(Engine):
    @torch.no_grad()
    def validation(self):
        
        pt_to_save={'arch': args.modelname,'state_dict': self.pretrain.module.imagemol.state_dict()}
        ft_to_save={'arch': args.modelname,'state_dict': self.pretrain.module.ftfc.state_dict()}
        rt_to_save={'arch': args.modelname,'state_dict': self.reweight.module.state_dict()}

        torch.save(pt_to_save, '{}/{}/model_pretrain_{}.pth'.format(args.save_dir,args.dataset,str(self.global_step)))
        torch.save(ft_to_save, '{}/{}/model_finetune_{}.pth'.format(args.save_dir,args.dataset,str(self.global_step)))
        torch.save(rt_to_save, '{}/{}/model_reweight_{}.pth'.format(args.save_dir,args.dataset,str(self.global_step)))


        weight=torch.softmax(self.reweight.module.weight,0)

        if args.wandb:
            wandb.log({"mg3c reweight value 1": weight[0]})
            wandb.log({"mg3c reweight value 2": weight[1]})
            wandb.log({"mg3c reweight value 3": weight[2]})
            wandb.log({"jpp reweight value": weight[3]})
            wandb.log({"mcl reweight value": weight[4]})

        return_dict = {
            "mg3c reweight value 1": weight[0],
            "mg3c reweight value 2": weight[1],
            "mg3c reweight value 3": weight[2],
            "jpp reweight value": weight[3],
            "mcl reweight value": weight[4]
        }
        return return_dict

if args.load_save:

    model = torch.load('{}/model_pretrain.pth'.format(args.save_dir)).to(device)
    model_reweight = torch.load('{}/model_reweight.pth'.format(args.save_dir)).to(device)
#Define Optimizers

# create optimizer
optimizer_pretrain = torch.optim.SGD(
    model.parameters(),
    lr=args.pretrain_lr,
    momentum=args.momentum,
    weight_decay=10 ** args.weight_decay,
)


optimizer_reweight = torch.optim.SGD(model_reweight.parameters(), lr=args.reweight_lr)

# Define configs
reweight_config = Config(type="darts", retain_graph=True)
pretrain_config = Config(type="darts",unroll_steps=args.unroll_steps, precision="fp16")
engine_config = EngineConfig(valid_step=args.val_freq, train_iters=args.iters, roll_back=False)

reweight = Reweighting(name="reweight", module=model_reweight,optimizer=optimizer_reweight,train_data_loader=eval_dataloader,config=reweight_config)
pretrain = Pretraining(name="pretrain",module=model_pretrain,optimizer=optimizer_pretrain,train_data_loader=combined_pretrain_dataloader, config=pretrain_config)


problems = [reweight, pretrain]

u2l = {reweight: [pretrain]}
l2u = {pretrain: [reweight]}
dependencies = {"u2l": u2l, "l2u": l2u}

engine = LBIEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()

