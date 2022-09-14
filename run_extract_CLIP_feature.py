import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

import clip

from datasets import build_dataset
import utils

def get_args():
     parser = argparse.ArgumentParser('VideoMAE evaluation script for video classification', add_help=False)
     parser.add_argument('--batch_size', default=64, type=int)

     # Model parameters
     parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                         help='Name of model to train')
     parser.add_argument('--tubelet_size', type=int, default= 2)
     parser.add_argument('--input_size', default=224, type=int,
                         help='videos input size')

     parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                         help='Dropout rate (default: 0.)')
     parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                         help='Attention dropout rate (default: 0.)')
     parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                         help='Drop path rate (default: 0.1)')

     parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
     parser.add_argument('--model_ema', action='store_true', default=False)
     parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
     parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

     # Evaluation parameters
     parser.add_argument('--crop_pct', type=float, default=None)
     parser.add_argument('--short_side_size', type=int, default=224)
     parser.add_argument('--test_num_segment', type=int, default=5)
     parser.add_argument('--test_num_crop', type=int, default=3)

     # Random Erase params
     parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                         help='Random erase prob (default: 0.25)')
     parser.add_argument('--remode', type=str, default='pixel',
                         help='Random erase mode (default: "pixel")')
     parser.add_argument('--recount', type=int, default=1,
                         help='Random erase count (default: 1)')
     parser.add_argument('--resplit', action='store_true', default=False,
                         help='Do not random erase first (clean) augmentation split')
     
     # Finetuning params
     parser.add_argument('--finetune', default='', help='finetune from checkpoint')
     parser.add_argument('--model_key', default='model|module', type=str)
     parser.add_argument('--model_prefix', default='', type=str)
     parser.add_argument('--init_scale', default=0.001, type=float)
     parser.add_argument('--use_mean_pooling', action='store_true')
     parser.set_defaults(use_mean_pooling=True)

     # Dataset parameters
     parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                         help='dataset path')
     parser.add_argument('--eval_data_path', default=None, type=str,
                         help='dataset path for evaluation')
     parser.add_argument('--nb_classes', default=400, type=int,
                         help='number of the classification types')
     parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
     parser.add_argument('--num_segments', type=int, default= 1)
     parser.add_argument('--num_frames', type=int, default= 16)
     parser.add_argument('--sampling_rate', type=int, default= 4)
     parser.add_argument('--data_set', default='Kinetics-400', choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51','image_folder'],
                         type=str, help='dataset')
     parser.add_argument('--output_dir', default='',
                         help='path where to save, empty for no saving')
     parser.add_argument('--log_dir', default=None,
                         help='path where to tensorboard log')
     parser.add_argument('--device', default='cuda',
                         help='device to use for training / testing')
     parser.add_argument('--seed', default=0, type=int)
     parser.add_argument('--resume', default='',
                         help='resume from checkpoint')
     parser.add_argument('--auto_resume', action='store_true')
     parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
     parser.set_defaults(auto_resume=True)

     parser.add_argument('--save_ckpt', action='store_true')
     parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
     parser.set_defaults(save_ckpt=True)

     parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                         help='start epoch')
     parser.add_argument('--eval', action='store_true',
                         help='Perform evaluation only')
     parser.add_argument('--dist_eval', action='store_true', default=False,
                         help='Enabling distributed evaluation')
     parser.add_argument('--extract', action='store_true',
                         help='Perform feature extract only')
     parser.add_argument('--extract_clip', action= 'store_true')
     parser.add_argument('--num_workers', default=10, type=int)
     parser.add_argument('--pin_mem', action='store_true',
                         help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
     parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
     parser.set_defaults(pin_mem=True)

     # distributed training parameters
     parser.add_argument('--world_size', default=1, type=int,
                         help='number of distributed processes')
     parser.add_argument('--local_rank', default=-1, type=int)
     parser.add_argument('--dist_on_itp', action='store_true')
     parser.add_argument('--dist_url', default='env://',
                         help='url used to set up distributed training')

     parser.add_argument('--enable_deepspeed', action='store_true', default=False)

     known_args, _ = parser.parse_known_args()
    
     if known_args.enable_deepspeed:
          try:
               import deepspeed
               from deepspeed import DeepSpeedConfig
               parser = deepspeed.add_config_arguments(parser)
               ds_init = deepspeed.initialize
          except:
               print("Please 'pip install deepspeed'")
               exit(0)
     else:
          ds_init = None
     return parser.parse_args(), ds_init

def main(args, ds_init):
     extract_path = '/data/dataset/something-something/extracted_clip_feature_ssv2/clip_768_ssv2'
     
     print(args)
     
     device = torch.device(args.device)
     
     # fix the seed for reproducibility
     seed = args.seed + utils.get_rank()
     torch.manual_seed(seed)
     np.random.seed(seed)
     
     cudnn.benchmark = True
     
     dataset_extract, _ = build_dataset(is_train=False, test_mode=False, args=args)
     
     num_tasks = utils.get_world_size()
     global_rank = utils.get_rank()
     

     
     if global_rank == 0 and args.log_dir is not None:
          os.makedirs(args.log_dir, exist_ok = True)
          log_writer = utils.TensorboardLogger(log_dir = args.log_dir)
     else:
          log_writer = None
          
     # define DataLoader
     dataset_loader_extract = torch.utils.data.DataLoader(
          dataset_extract,
          batch_size = args.batch_size,
          num_workers = args.num_workers,
          pin_memory = args.pin_mem,
          drop_last = False
     )
     
     # create model.
     model, _ = clip.load("ViT-B/32", device=device)
     model.visual.proj = None
     
     # engine_for_finetuning > validation_one_epoch 참조하여 작성.
     with torch.no_grad():
          metric_logger = utils.MetricLogger(delimiter="  ")
          header = 'extract_clip'
          
          for batch in metric_logger.log_every(dataset_loader_extract, 10, header):
               videos = batch[0]
               # center frame만 골라낸다.
               center_frame = videos[:, :, 8, :, :].to(device)
               file_names = batch[2]
               
               #comput output
               output = model.encode_image(center_frame)
               output = output.cpu().numpy()
               for i, one_file in enumerate(file_names):
                    save_filename = os.path.join(extract_path, one_file)
                    np.save(save_filename + '.npy', output[i])
                    

if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)