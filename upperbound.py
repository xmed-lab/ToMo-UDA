from __future__ import  absolute_import
import os

import numpy as np
import matplotlib
from tqdm import tqdm
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from data.dataset import inverse_normalize
from data.fetus_dataset import fetus_Dataset, collate_fn, annnotations_convert

from utils.ub_config import opt
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from utils.boxlist import BoxList
from utils.gpu_tools import get_world_size, get_global_rank, get_local_rank, get_master_ip

from utils.distributed import get_rank, synchronize, reduce_loss_dict, DistributedSampler, all_gather
from utils.graph_config import _C as graph_opt
from utils.build_opt import make_optimizer, make_lr_scheduler
from utils.vis_bbox import plot_bboxes

from model.topograph_net import Topograph
from model.graph_matching import build_graph_matching_head
from model.discriminator import Discriminator

import resource
import wandb
from skimage import exposure

import warnings
warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


class Trainer():
    def __init__(self, opt):
        self.opt = opt

        self.opt.n_class = len(annnotations_convert[opt.slices[0]]) + 1
        graph_opt.MODEL.FCOS.NUM_CLASSES = len(annnotations_convert[opt.slices[0]]) + 1
        
        print('Load Fetus Dataset')
        train_source_set = fetus_Dataset(self.opt, operation='train')
        self.train_source_dataloader = DataLoader(train_source_set,
                                        collate_fn = collate_fn(opt),
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=self.opt.num_workers)

        # train_target_set = fetus_Dataset(self.opt, operation='train', domain='Target')
        # self.train_target_dataloader = DataLoader(train_target_set,
        #                                 collate_fn = collate_fn(opt),
        #                                 batch_size=opt.batch_size,
        #                                 shuffle=True,
        #                                 drop_last=True,
        #                                 num_workers=self.opt.num_workers)

        vaildset = fetus_Dataset(self.opt, operation='valid')
        self.vaild_dataloader = DataLoader(vaildset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,)

        testset  = fetus_Dataset(self.opt, operation='test')
        self.test_dataloader = DataLoader(testset,
                                        collate_fn = collate_fn(opt),
                                        batch_size=1,
                                        num_workers=self.opt.test_num_workers,
                                        shuffle=False,)

        print('Build BackBone Network & Graph Matching Module')
        self.model = Topograph(self.opt, Topograph_m=True).to(device=opt.device)
        # self.graph_matching = build_graph_matching_head(graph_opt, self.opt.out_channel).to(device=opt.device)
        # # discriminator
        # if opt.discriminator:
        #     self.dis_dict = dict()
        #     self.dis_dict['dis_p2'] = Discriminator(grad_reverse_lambda=0.1) # grad_reverse_lambda=0.02
        #     self.dis_dict['dis_p3'] = Discriminator(grad_reverse_lambda=0.1)
        #     self.dis_dict['dis_p4'] = Discriminator(grad_reverse_lambda=0.1)
        #     self.dis_dict['dis_p5'] = Discriminator(grad_reverse_lambda=0.1)

        print('Model Construct Completed')

        print('Build Optimizer & Scheduler for BackBone and Graph Matching')
        self.optimizer = {}
        self.scheduler = {}
        
        self.optimizer["backbone"] = make_optimizer(graph_opt, self.model, name='backbone')
        # self.optimizer["middle_head"] = make_optimizer(graph_opt, self.graph_matching, name='backbone')
        self.scheduler["backbone"] = make_lr_scheduler(graph_opt, self.optimizer["backbone"], name='middle_head')
        # self.scheduler["middle_head"] = make_lr_scheduler(graph_opt, self.optimizer["middle_head"], name='middle_head')

        #discriminator
        # if opt.discriminator:
        #     self.optimizer['Dis_P2'] = make_optimizer(graph_opt, self.dis_dict['dis_p2'], name='discriminator')
        #     self.optimizer['Dis_P3'] = make_optimizer(graph_opt, self.dis_dict['dis_p3'], name='discriminator')
        #     self.optimizer['Dis_P4'] = make_optimizer(graph_opt, self.dis_dict['dis_p4'], name='discriminator')
        #     self.optimizer['Dis_P5'] = make_optimizer(graph_opt, self.dis_dict['dis_p5'], name='discriminator')

        #     self.scheduler['Dis_P2'] = make_lr_scheduler(graph_opt, self.optimizer['Dis_P2'], name='discriminator')
        #     self.scheduler['Dis_P3'] = make_lr_scheduler(graph_opt, self.optimizer['Dis_P3'], name='discriminator')
        #     self.scheduler['Dis_P4'] = make_lr_scheduler(graph_opt, self.optimizer['Dis_P3'], name='discriminator')
        #     self.scheduler['Dis_P5'] = make_lr_scheduler(graph_opt, self.optimizer['Dis_P3'], name='discriminator')

        #     for key in self.dis_dict.keys():
        #         self.dis_dict[key].to(device=opt.device)

        # self.writer = SummaryWriter()

        if self.opt.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.opt.local_rank],
                output_device=self.opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,)

            self.graph_matching = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.graph_matching)
            self.graph_matching = torch.nn.parallel.DistributedDataParallel(
                self.graph_matching,
                device_ids=[self.opt.local_rank],
                output_device=self.opt.local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True,)

    def train(self):
        if self.opt.load_path:
            self.load(self.opt.load_path)
            print('load pretrained model from %s' % self.opt.load_path)

        best_map = 0
        dis_loss = {}
        batch_step = 0
        lr_ = self.opt.lr
        for epoch in range(self.opt.epoch):
            self.model.train()
            # self.graph_matching.train()

            # len_source = len(self.train_source_dataloader)
            # len_target = len(self.train_target_dataloader)
            # max_len = max(len_source, len_target)

            # source_iter = iter(self.train_source_dataloader)
            # target_iter = iter(self.train_target_dataloader)

            # for _ in tqdm(range(max_len)):
            for _, (imgs_src, targets_src, _) in enumerate(tqdm(self.train_source_dataloader)):

                # try:
                #     imgs_src, targets_src, _ = next(source_iter)
                # except StopIteration:
                #     source_iter = iter(self.train_source_dataloader)
                #     imgs_src, targets_src, _ = next(source_iter)

                # try:
                #     imgs_tgt, _, _ = next(target_iter)
                # except StopIteration:
                #     target_iter = iter(self.train_target_dataloader)
                #     imgs_tgt, _, _ = next(target_iter)

                # # imgs_src, targets_src, _ = next(train_source_dataloader)
                # # imgs_tgt, _, _ = next(train_target_dataloader)
                targets_src = [target.to(device=opt.device) for target in targets_src]

                # # sp Transfer
                # imgs_sp_s = [
                #     torch.tensor(
                #         exposure.match_histograms(
                #             img.permute(1, 2, 0).numpy(), 
                #             img_target.permute(1, 2, 0).numpy()
                #         )
                #     ).permute(2, 0, 1)
                #     for img, img_target in zip(imgs_src.tensors, imgs_tgt.tensors)
                # ]

                # imgs_src = torch.stack(imgs_sp_s, dim=0).float()
                
                (features_src, _, _, _), _, losses = \
                    self.model(imgs_src.tensors.to(device=opt.device), image_sizes=None, targets=targets_src, train=True, domain='Source')
                # (features_tgt, cls_pred_tgt, box_pred_tgt, center_pred_tgt), _, _ = \
                #     self.model(imgs_tgt.tensors.to(device=opt.device), image_sizes=None, targets=None, train=True, domain='Target')
                
                # score_maps_tgt = self.model._forward_target(cls_pred_tgt, box_pred_tgt, center_pred_tgt)

                # (_, _), middle_head_loss, loss_cont, loss_sink = \
                #     self.graph_matching(None, (features_src, features_tgt), targets=targets_src, score_maps=score_maps_tgt)

                # discriminator
                # if self.opt.discriminator:
                #     for layer, layer_name in enumerate(['p2', 'p3', 'p4', 'p5']):
                #             dis_loss["loss_adv_%s" % layer_name] = \
                #                 0.1 * self.dis_dict["dis_%s" % layer_name]((features_src[layer],features_tgt[layer])) # 
                
                loss_cls = losses['loss_cls'].mean()
                loss_box = losses['loss_box'].mean()
                loss_center = losses['loss_center'].mean()
                backbone_loss = loss_cls + loss_box + loss_center
                # loss_matching = sum(loss for loss in middle_head_loss.values())

                # if self.opt.discriminator:
                #     dis_losses = sum(loss for loss in dis_loss.values())
                #     overall_loss = backbone_loss + loss_matching + loss_cont + loss_sink + dis_losses

                # else:
                #     overall_loss = backbone_loss + loss_matching + loss_cont + loss_sink
                
                # tensorboard
                # self.writer.add_scalar('Losses/backbone_loss', backbone_loss.item(), epoch)
                # self.writer.add_scalar('Losses/graph_matching_loss', loss_matching.item(), epoch)
                # if self.opt.discriminator:
                #     self.writer.add_scalar('Losses/discriminator_loss', dis_losses.item(), epoch)
                # self.writer.add_scalar('Losses/contractive_loss', loss_cont.item(), epoch)
                # self.writer.add_scalar('Losses/sinkhorn_dis', loss_sink.item(), epoch)
                # self.writer.add_scalar('Losses/overall_loss', overall_loss.item(), epoch)

                for opt_k in self.optimizer:
                    self.optimizer[opt_k].zero_grad()
                
                backbone_loss.backward()

                for opt_k in self.optimizer:
                    self.optimizer[opt_k].step()

            eval_result = self.eval(self.vaild_dataloader, test_num=self.opt.test_num)

            print(f"backbone_loss:{backbone_loss.item()}")
            # print(f'graph_matching_loss:{loss_matching.item()}')
            # if self.opt.discriminator:
            #     print(f'discriminator_loss:{dis_losses.item()}')
            # print(f'cont_loss:{loss_cont.item()}')
            # print(f'sink_loss:{loss_sink.item()}')
            log_info = 'epoch:{}, map:{},loss:{}'.format(str(epoch),
                                                         str(round(eval_result['map'], 4)),
                                                         str(backbone_loss.item()))
            print(log_info)
            # self.writer.add_scalar('mAP:', round(eval_result['map'], 4), epoch)


            # Update optimizers with scheduler
            for scheduler_k in self.scheduler:
                self.scheduler[scheduler_k].step()
            
            if eval_result['map'] > best_map and epoch >= 30:
                best_map = eval_result['map']
                # if best_map > 0.5:
                best_path = self.save(best_map=best_map)

            # self.writer.add_scalar('mAP', eval_result['map'], epoch)

            if epoch == opt.epoch-1: 
                print(f"Testing: load best_path:{best_path}")
                self.load(best_path)
                test_result = self.eval(self.test_dataloader, test_num=self.opt.test_num)
                log_info = 'final test ---> epoch:{}, map:{},loss:{}'.format(str(epoch),
                                                                             str(test_result['map']),
                                                                             str(backbone_loss.item()))
                print(log_info)
                break

        # self.writer.close()

    def accumulate_predictions(self, predictions):
        all_predictions = all_gather(predictions)

        if get_rank() != 0:
            return

        predictions = {}

        for p in all_predictions:
            predictions.update(p)

        ids = list(sorted(predictions.keys()))

        if len(ids) != ids[-1] + 1:
            print('Evaluation results is not contiguous')

        predictions = [predictions[i] for i in ids]

        return predictions

    @torch.no_grad()
    def eval(self, dataloader, path, gt=False, test_num=10000):
        self.model.eval()
        pred_bboxes, pred_labels, pred_scores = list(), list(), list()
        gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
        for ids, (imgs, gt_targets, ids) in tqdm(enumerate(dataloader)):

            preds = {}
            imgs = imgs.tensors.to(device=opt.device)

            gt_targets = [target.to('cpu') for target in gt_targets]

            pred, _ = self.model(imgs, imgs.shape[-2:], train=False)
            pred = [p.to('cpu') for p in pred]
            preds = pred

            for idx, pred in enumerate(preds):
                _pred_bboxes = pred.box.numpy()
                _pred_labels = pred.fields['labels'].numpy()
                _pred_scores = pred.fields['scores'].numpy()
                _gt_bboxes_ = gt_targets[idx].box.numpy()
                _gt_labels_ = gt_targets[idx].fields['labels'].numpy()
                if self.opt.draw_bbox:
                    plot_bboxes(imgs, _pred_bboxes, _pred_labels, _pred_scores, _gt_bboxes_, _gt_labels_, path, ids, gt)
            if _pred_bboxes.shape[0] == 0:
                continue
            else:
                pred_bboxes += [_pred_bboxes]
                pred_labels += [_pred_labels]
                pred_scores += [_pred_scores]

                gt_bboxes += [_gt_bboxes_]
                gt_labels += [_gt_labels_]
                # gt_difficults.append(gt_difficults_)

            if ids == test_num: break

        gt_difficults = None
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults,
            use_07_metric=True)
        return result

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.model.state_dict()
        save_dict['config'] = opt._state_dict()

        if save_optimizer:
            for opt_k in self.optimizer:
                save_dict['optimizer'][opt_k] = self.optimizer[opt_k].state_dict()

        if save_path is None:
            # timestr = time.strftime('%m%d%H%M')
            save_path = f'checkpoints/ub_{self.opt.selected_source_hospital[0]}_{self.opt.slices[0]}'
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False,):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.model.load_state_dict(state_dict)
            return self
        if parse_opt:
            self.opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])

def main(rank, opt):

    try:
        opt.local_rank
    except AttributeError:
        opt.global_rank = rank
        opt.local_rank = opt.enable_GPUs_id[rank]
    else:
        if opt.distributed:
            opt.global_rank = rank
            opt.local_rank = opt.enable_GPUs_id[rank]

    if opt.distributed:
        torch.cuda.set_device(int(opt.local_rank))
        torch.distributed.init_process_group(backend='nccl',
                                             init_method=opt.init_method,
                                             world_size=opt.world_size,
                                             rank=opt.global_rank,
                                             group_name='mtorch'
                                             )

        print('using GPU {}-{} for training'.format(
            int(opt.global_rank), int(opt.local_rank)
            ))

        if opt.local_rank == opt.enable_GPUs_id[0]:
            wandb_init()

    if torch.cuda.is_available(): 
        opt.device = torch.device("cuda:{}".format(opt.local_rank))
    else: 
        opt.device = 'cpu'
    
    Train_ = Trainer(opt)
    Train_.train()

if __name__ == '__main__':

    # setting distributed configurations
    opt.world_size = len(opt.enable_GPUs_id)
    opt.init_method = f"tcp://{get_master_ip()}:{23455}"
    opt.distributed = True if opt.world_size > 1 else False

    # setup distributed parallel training environments
    if get_master_ip() == "127.0.0.1" and opt.distributed:
        # manually launch distributed processes 
        torch.multiprocessing.spawn(main, nprocs=opt.world_size, args=(opt,))
    else:
        # multiple processes have been launched by openmpi
        opt.local_rank = opt.enable_GPUs_id[0]
        opt.global_rank = opt.enable_GPUs_id[0]

        main(opt.local_rank, opt)
