import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import logging
from typing import Callable, Sequence, Tuple, Union

import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils import ramps, losses, test_patch, info_nce, val_2d
from dataloaders.dataset import *
from networks.net_factory import net_factory


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

#I am a donkey

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='ACDC', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='fbanet', help='exp_name')
parser.add_argument('--model', type=str,  default='fbanet_2d', help='model_name')
parser.add_argument('--dim', type=int,  default=128, help='dim of MLP')
parser.add_argument('--max_iteration', type=int,  default=30000, help='maximum iteration to train')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--labeled_bs', type=int, default=8, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=7, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_{}_labeled_batch{}/{}".format(args.dataset_name, args.exp, args.dim, args.labelnum, args.batch_size, args.model)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

args.root_path = args.root_path+'data/ACDC'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    model = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes, mode="train", dim=args.dim, dataset=args.dataset_name)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split='train',
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split='val')

        
    labelnum = args.labelnum  
    total_slices = len(db_train)
    label_slice = patients_to_slices(args.root_path, labelnum)
    labeled_idxs = list(range(0, label_slice))
    unlabeled_idxs = list(range(label_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=32, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=32)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    consistency_criterion = losses.mse_loss
    dice_loss = losses.DiceLoss(args.num_classes)
    min_loss = losses.SimMinLoss()
    max_loss = losses.SimMaxLoss()

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_batch = volume_batch[labeled_bs:]

            model.train()
            feats, outputs = model(volume_batch)
            num_outputs = len(outputs)

            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

            loss_seg_dice = 0 
            loss_contra = 0
            
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs,...]
                y_prob = F.softmax(y, dim=1)
                loss_seg_dice += dice_loss(y_prob, label_batch[:labeled_bs].unsqueeze(1))

                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)

                feat = feats[idx]
                for n in range(args.num_classes):
                    pos = feat[:, n]
                    neg = torch.stack([feat[:, i] for i in range(args.num_classes) if i != n], dim=1)
                    loss_contra += max_loss(pos)
                    for cls in range(neg.shape[1]):
                        loss_contra += min_loss(pos, neg[:, cls])
            loss_contra /= (args.num_classes * (args.num_classes - 1))

            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
            
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            loss_supervised = loss_seg_dice

            loss = loss_supervised + consistency_weight * loss_consist + 1 * loss_contra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)
                
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=args.num_classes, contra=True, version=2)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(args.num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                dice_sample = np.mean(metric_list, axis=0)[0]
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        logging.info('Epoch %d : loss : %03f, loss_super: %03f, loss_cosist: %03f, loss_contra: %03f' % (epoch_num, loss, loss_supervised, loss_consist, loss_contra))
        
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
