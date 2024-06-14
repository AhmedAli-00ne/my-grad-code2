import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from scipy.stats import pearsonr as corr

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test
from models import build_model

from utils import *

from dataset_algonauts import fetch_dataloader

import code
import pprint

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    
    ## algonauts params
    parser.add_argument('--subj', default=1, type=int)  # 5 is a good test sub
    parser.add_argument('--run', default=1, type=int)  # 5 is a good test sub
    parser.add_argument('--data_dir', default='algonauts_2023_challenge_data/', type=str)
    parser.add_argument('--parent_submission_dir', default='algonauts_2023_challenge_submission/', type=str)
    
    parser.add_argument('--saved_feats', default=None, type=str) #'dinov2q'
    parser.add_argument('--saved_feats_dir', default='../../algonauts_image_features/', type=str) 
    
    parser.add_argument('--decoder_arch', default='transformer', type=str) #'dinov2q'
    parser.add_argument('--readout_res', default='streams_inc', type=str)   
    # [streams_inc, visuals, bodies, faces, places, words, hemis]

    # Model parameters
    
    #'../results/detr_grouping_256_2/checkpoint_0.50300_1.pth'
    parser.add_argument('--resume', default=None, help='resume from checkpoint')

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    parser.add_argument('--pretrained', type=str, default=None,
                        help="all the weights that can be, will be initialized from this pretrained model") 
                            #'../../pretrained/detr_r50.pth'
    
#     parser.add_argument('--pretrained_params', type=str, default='backbone input_projection encoder decoder',
#                         help="To limit the scope of what can be initialized using the pretrained model") 
    
    parser.add_argument('--frozen_params', type=str, default='backbone input_proj',
                        help="These components will not be retrained")    
    
    # * Backbone
    parser.add_argument('--backbone', default='dinov2', type=str,
                        help="Name of the convolutional backbone to use")  #resnet50
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")  #256  #868 (100+768) 
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=16, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=16, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    
    parser.add_argument('--enc_output_layer', default=1, type=int,
                    help="Specify the encoder layer that provides the encoder output. If None, will be the last layer")
    
    parser.add_argument('--output_layer', default='backbone', type=str,
                    help="If no encoder (enc_layers = 0), what to use to feed to the linear classifiers; input_proj or backbone")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=1, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    
    # other params
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--class_loss_coef', default=1, type=float)
    parser.add_argument('--cosine_loss_coef', default=0, type=float)
    
    parser.add_argument('--task', default = 'algonauts') 
    parser.add_argument('--distributed', default=0, type=int)
    
    parser.add_argument('--wandb', default='subj03_run3', type=str)

    # dataset parameters
    parser.add_argument('--dataset_grouping_dir', default='./datasets/dataset_grouping/')
    
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default ='../../coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./results/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--save_model', default=0, type=int) 

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    device = torch.device(args.device)
    
    args.subj = format(args.subj,'02')
    args.data_dir = os.path.join(args.data_dir, 'subj'+args.subj)
    args.subject_submission_dir = os.path.join(args.parent_submission_dir, 'subj'+args.subj)
    save_dir = args.output_dir + 'detr_dino_'+str(args.enc_output_layer) + '_' + args.readout_res + '_' + str(args.num_queries) + '/' + str(args.subj) + '/run' + str(args.run) + '/'
    args.save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(args.subject_submission_dir):
        os.makedirs(args.subject_submission_dir)
        
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
            allow_pickle=True).item())
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))
    
    if args.readout_res == 'visuals':
        args.rois_ind = 0
        args.num_queries = 16   # 2*len(roi_name_maps[args.rois_ind])
        
    elif args.readout_res == 'bodies':
        args.rois_ind = 1
        args.num_queries = 16 # 10
        
    elif args.readout_res == 'faces':
        args.rois_ind = 2
        args.num_queries = 16 #12
            
    elif args.readout_res == 'places':
        args.rois_ind = 3
        args.num_queries = 16 #8
                
    elif args.readout_res == 'words':
        args.rois_ind = 4
        args.num_queries = 16 # 12
        
    elif args.readout_res == 'streams' or args.readout_res == 'streams_inc':
        args.rois_ind = 5
        args.num_queries = 16
            
    elif args.readout_res == 'hemis':
        args.rois_ind = 5
        args.num_queries = 2
    
    args.roi_nums = len(roi_name_maps[args.rois_ind])

    lh_rois = torch.tensor(lh_challenge_rois[args.rois_ind]).to(args.device)  # -1
    rh_rois = torch.tensor(rh_challenge_rois[args.rois_ind]).to(args.device)  # -1

    lh_challenge_rois_s = []
    rh_challenge_rois_s = []
    for i in range(args.roi_nums):
        lh_challenge_rois_s.append(torch.where(lh_rois == i, 1, 0))
        rh_challenge_rois_s.append(torch.where(rh_rois == i, 1, 0))

    lh_challenge_rois_s = torch.vstack(lh_challenge_rois_s)
    rh_challenge_rois_s = torch.vstack(rh_challenge_rois_s)

        
    args.lh_vs = len(lh_challenge_rois_s[args.rois_ind])
    args.rh_vs = len(rh_challenge_rois_s[args.rois_ind])
    
    model, criterion = build_model(args)
    model.to(device)
    
    args.best_val_acc = 0
    
    param_dicts = [ { "params" : [ p for n , p in model.named_parameters() if p.requires_grad]}, ] 
        
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                              weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    pretrained_params = []
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            
        pretrained_dict = checkpoint['model']
        model.load_state_dict(pretrained_dict)
        
        args.best_val_acc = vars(checkpoint['args'])['best_val_acc'] #checkpoint['val_acc'] #or read it from the   
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            
            train_params = checkpoint['train_params']
            param_dicts = [ { "params" : [ p for n , p in model.named_parameters() if n in train_params ]}, ] 
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                          weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
        
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        print('train_params_resume', train_params)
    elif args.pretrained:
        # gather the list of parameters that can be initialzed with the pre-trained model
        
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained) 
        pretrained_dict = pretrained_dict['model'] 
        
        for n , p in model.named_parameters(): 
#             if (args.pretrained_params):
#                 for component in args.pretrained_params.split(' '):
            if ((n in pretrained_dict.keys()) and (p.shape == pretrained_dict[n].shape)): 
                pretrained_params.append(n) 

        print('\npretrained_params', pretrained_params)  

        pretrained_dict = { k : v for k , v in pretrained_dict.items() if k in pretrained_params } 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict) 
    
        # gather the list of parameters that will not change after initializaiton
        frozen_params = [] 
        for n , p in model.named_parameters(): 

            for component in args.frozen_params.split(' '):
                if (component in n and n in pretrained_params): 
                    frozen_params.append(n) 
                    p.requires_grad = False
                    
                    # todo should I jsut put p.requires_grad = 0 and that may be saved with resume

        print('\nfrozen_params', frozen_params)  
    
        param_dicts = [ 
            { "params" : [ p for n , p in model.named_parameters() if p.requires_grad]}, ]  #n not in frozen_params and 
    
        train_params = [ n for n , p in model.named_parameters() if p.requires_grad ]  # n not in frozen_params and

        print('\ntrain_params', train_params)
        
        

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    else:
        
        param_dicts = [ 
            { "params" : [ p for n , p in model.named_parameters() if p.requires_grad]}, ]  #n not in frozen_params and 
    
        train_params = [ n for n , p in model.named_parameters() if p.requires_grad ]  # n not in frozen_params and

        print('\ntrain_params', train_params)

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    train_dataloader, val_dataloader = fetch_dataloader(args, args.batch_size, train='train')
    test_dataloader = fetch_dataloader(args, args.batch_size, train='test')

        

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)


    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        print("samo")
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return



    with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
        pprint.pprint(args.__dict__, f, sort_dicts=False)
        
        
    with open(os.path.join(args.save_dir, 'val_results.txt'), 'w') as f:
        f.write(f'validation results: \n') 
        
        
    print("Start training")
    start_time = time.time()
    args.val_mean_corr = -100
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, train_dataloader, optimizer, args.device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        
        lh_fmri_val_pred, rh_fmri_val_pred, lh_fmri_val, rh_fmri_val, val_loss = evaluate(model, criterion, val_dataloader, args, lh_challenge_rois_s, rh_challenge_rois_s)
        
        
        # Empty correlation array of shape: (LH vertices)
        lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh_fmri_val_pred.shape[1])):
            lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

        # Empty correlation array of shape: (RH vertices)
        rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh_fmri_val_pred.shape[1])):
            rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]
        
        # Select the correlation results vertices of each ROI
        roi_names = []
        lh_roi_correlation = []
        rh_roi_correlation = []
        for r1 in range(len(lh_challenge_rois)):
            for r2 in roi_name_maps[r1].items():
                if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                    roi_names.append(r2[1])
                    lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                    rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                    lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                    rh_roi_correlation.append(rh_correlation[rh_roi_idx])
        roi_names.append('All vertices')
        lh_roi_correlation.append(lh_correlation)
        rh_roi_correlation.append(rh_correlation)
        

        # Create the plot
        lh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(lh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
            for r in range(len(lh_roi_correlation))]
        rh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(rh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
            for r in range(len(rh_roi_correlation))]
        
        val_mean_corr = (lh_mean_roi_correlation[-1] + rh_mean_roi_correlation[-1]) / 2
        
        print('val_mean_corr:', val_mean_corr ) 
        print('shape of rh_fmri_val_pred', rh_fmri_val_pred.shape)



        if args.output_dir:
            
            if (val_mean_corr > args.val_mean_corr):  
                args.val_mean_corr = val_mean_corr                
                checkpoint_paths = [save_dir + 'checkpoint_'+'.pth'] 
        
                with open(os.path.join(args.save_dir, 'val_results.txt'), 'a') as f:
                    f.write(f'epoch {epoch}, val_mean_corr: {val_mean_corr}, val_loss: {val_loss} \n') 
        
                if args.save_model:
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'train_params' : train_params,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
                    
               
                np.save(args.save_dir+'lh_fmri_val_pred.npy', lh_fmri_val_pred)
                np.save(args.save_dir+'rh_fmri_val_pred.npy', rh_fmri_val_pred)
                
                
                lh_fmri_test_pred, rh_fmri_test_pred = test(model, criterion, test_dataloader, args, lh_challenge_rois_s, rh_challenge_rois_s)
                
                lh_fmri_test_pred = lh_fmri_test_pred.astype(np.float32)
                rh_fmri_test_pred = rh_fmri_test_pred.astype(np.float32)
                
                np.save(args.save_dir+'/lh_pred_test.npy', lh_fmri_test_pred)
                np.save(args.save_dir+'/rh_pred_test.npy', rh_fmri_test_pred)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)