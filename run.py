import argparse
import os
import torch
from exp.exp_rnn import Exp_RNN
from exp.exp_DANet import Exp_AT
from exp.exp_line import Exp_myline
from exp.exp_nbeats import Exp_Nbeats
import random
import numpy as np
import argparse
import os
import argparse
import torch
import random

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Time Series Forecasting Configuration')
parser.add_argument('--model_type', type=int, default=1, help='rnn:1,line:2,atnet:3,baseline:4')
parser.add_argument('--is_training', type=int, default=1, help='training mode')
parser.add_argument('--train_only', action='store_true', help='perform training on full input dataset without validation and testing')
parser.add_argument('--need_test', type=int, default=1, help='need to perform testing')
parser.add_argument('--model_id', type=str, default='DLINE_H', help='model identifier')
parser.add_argument('--model', type=str, default='DLINE', help='model type')
parser.add_argument('--do_predict', type=int, default=0, help='perform prediction')
parser.add_argument('--output_attention', type=int, default=0, help='output attention')

parser.add_argument('--data', type=str, default='parking_h', help='data set')
parser.add_argument('--data_path', type=str, default='./data/parking_data_h_new.csv', help='path to data file')
parser.add_argument('--features', type=str, default='S', help='feature set')
parser.add_argument('--target', type=str, default='cars_in_park', help='target feature')
parser.add_argument('--freq', type=str, default='h', help='frequency of data')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='path to save checkpoints')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--begin_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=53)
parser.add_argument('--test_index', type=int, default=47)

parser.add_argument('--seq_len', type=int, default=24*7, help='sequence length')
parser.add_argument('--label_len', type=int, default=24*30, help='label length')
parser.add_argument('--pred_len', type=int, default=24*30, help='prediction length')

parser.add_argument('--individual', action='store_true', help='use individual model')
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')

parser.add_argument('--method', type=str, default="adam", help='input size')
parser.add_argument('--max_grad_norm', type=int, default=100, help='griedent clip')
parser.add_argument('--lr_decay', type=int, default=0.99, help='output size')
parser.add_argument('--start_decay_at', type=int, default=100, help='decay epoch')
parser.add_argument('--set_loss', type=int, default=None, help='decay start loss')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')

parser.add_argument('--input_size', type=int, default=1, help='input size')
parser.add_argument('--output_size', type=int, default=1, help='output size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer size')
parser.add_argument('--hidCNN', type=int, default=100, help='hidden CNN size')
parser.add_argument('--CNN_kernel', type=int, default=6, help='CNN kernel size')
parser.add_argument('--highway_window', type=int, default=24, help='highway window size')
parser.add_argument('--clip', type=int, default=10, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--skip', type=int, default=24, help='skip length')
parser.add_argument('--hidSkip', type=int, default=5, help='hidden skip size')
parser.add_argument('--isRNN', type=int, default=1, help='use RNN')
parser.add_argument('--output_fun', type=str, default=None, help='output function')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
parser.add_argument('--itr', type=int, default=1, help='iteration count')
parser.add_argument('--train_epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--patience', type=int, default=7, help='patience for early stopping')
parser.add_argument('--des', type=str, default='test_100', help='description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')

parser.add_argument('--use_gpu', type=int, default=1, help='use GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--use_multi_gpu', default=0, help='use multiple GPUs')
parser.add_argument('--devices', type=str, default='0,1,2', help='device IDs for multiple GPU usage')
parser.add_argument('--test_flop',type=int, default=0, help='test FLOPs')

parser.add_argument('--random_size', type=int, default=1200, help='random size,>pred_len+seq_len')
parser.add_argument('--seq_number', type=int, default=47, help='seq number,namely column of train data')
parser.add_argument('--stride', type=int, default=10, help='stride')


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
if args.model_type==1:
    Exp = Exp_RNN
# if args.model_type==2:
#     Exp = Exp_Main
if args.model_type==3:
    Exp = Exp_AT
if args.model_type == 4:
    Exp = Exp_myline
if args.model_type == 5:
    Exp = Exp_Nbeats
if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            # args.d_model,
            # args.n_heads,
            # args.e_layers,
            # args.d_layers,
            # args.d_ff,
            # args.factor,
            args.embed,
            # args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if args.need_test:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}_{}_{}'.format(args.model_id,
                                                               args.model,
                                                               args.data,
                                                               args.features,
                                                               args.seq_len,
                                                               args.label_len,
                                                               args.pred_len,
                                                               # args.d_model,
                                                               # args.n_heads,
                                                               # args.e_layers,
                                                               # args.d_layers,
                                                               # args.d_ff,
                                                               # args.factor,
                                                               args.embed,
                                                               # args.distil,
                                                               args.des, ii)

    exp = Exp(args)  # set experiments

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
