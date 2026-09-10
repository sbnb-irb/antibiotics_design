import os
import sys

current_directory = os.getcwd()

sys.path.append('/scr/hvae_training/')
import utils
import hgraph
from hgraph import HierAE, common_atom_vocab, PairVocab

import gc
import argparse
import numpy as np
import pandas as pd
import math, random
from tqdm import tqdm
import json
import logging
import traceback
from datetime import datetime
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

#param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

def load_namespace_from_json(filepath: str,extra_args={}) -> SimpleNamespace:
    """Load JSON into a SimpleNamespace."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    for k,v in extra_args.items():
        data[k] = v
    return SimpleNamespace(**data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--atom_vocab', default=common_atom_vocab)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--trained_model',type=str,default="")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_config', type=str, default="")
    #
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--hidden_size', type=int, default=250)
    parser.add_argument('--embed_size', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--latent_size', type=int, default=32)
    parser.add_argument('--depthT', type=int, default=15)
    parser.add_argument('--depthG', type=int, default=15)
    parser.add_argument('--diterT', type=int, default=1)
    parser.add_argument('--diterG', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    #
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--nsample', type=int, default=-1)
    parser.add_argument('--print_iter', type=int, default=10)
    parser.add_argument('--save_total_steps', type=int, default=5000)
    parser.add_argument('--log_name', type=str, default="train_log")
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--save_at_end', type=int, default=-1)  # Save a ckpt at the end
    parser.add_argument('--print_inner_iter', type=int, default=-1)
    parser.add_argument('--loss_increase_tolerance', type=int, default=7)
    parser.add_argument('--save_best_loss', type=int, default=1)
    parser.add_argument('--best_loss', type=float, default=1e10)
    parser.add_argument('--smiles_are_safe', type=int, default=0)
    parser.add_argument('--only_get_safe_smiles', type=int, default=0)
    parser.add_argument('--save_safe_smiles_file', type=str, default="safe_smiles.csv")    
    #
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.save_dir,exist_ok=True)
    #Set up logging
    print("Setting logger")
    start_time_obj = datetime.now()
    time_now = start_time_obj.strftime("%Y%m%d-%H%M%S")
    _log_file = "{}_{}.log".format(args.log_name,time_now)
    logging.basicConfig(filename=_log_file, filemode='a', format='%(name)s - %(levelname)s - %(message)s',level=10)
    def log_msg(msg,ref_time=start_time_obj):
        time_now_obj = datetime.now()
        time_now =time_now_obj.strftime("%Y%m%d-%H%M%S")
        if ref_time is not None:
            delta_time = time_now_obj - ref_time
            total_seconds = int(delta_time.total_seconds())
            days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
            hours, remainder = divmod(remainder, 3600)      # 3600 seconds in an hour
            minutes, seconds = divmod(remainder, 60)        # 60 seconds in a minute
            delta_time_str = f"{days:02}:{hours:02}:{minutes:02}:{seconds:02}"
            _m = "{}: {}.".format(delta_time_str,msg)
        else:
            _m = "{}: {}.".format(time_now,msg)
        print(_m)
        logging.info(_m)
    log_msg("Finished args parsing")
    log_msg(args)

    #Load data
    train_smiles = list(pd.read_csv(args.smiles_file,header=None).iloc[:,0].values)
    vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
    args.vocab = PairVocab(vocab)
    log_msg("vocab Paired")

    #Load a subset for testing
    if args.nsample>0:
        n = len(train_smiles)
        n = min(args.nsample,n)
        train_smiles = train_smiles[:n]
    
    #If smiles are safe do not loop for valid smiles
    if args.smiles_are_safe==1:
        log_msg("Loading pre validated files. No Filtering.")
        dataset = hgraph.HAEMoleculeDatasetSafe(train_smiles, args.vocab, args.atom_vocab, args.batch_size)
    else:
        log_msg("Creating new dataset")
        dataset = hgraph.HAEMoleculeDatasetNew(train_smiles, args.vocab, args.atom_vocab, args.batch_size)
        
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x:x[0], shuffle=False, num_workers=args.cpus)
    total_batches = len(dataloader)
    log_msg(f"total batches: {total_batches}")
    if args.only_get_safe_smiles==1:
        log_msg("Collecting and saving safe SMILES")
        _osmiles = []
        for batch_data in tqdm(dataloader, desc="Collecting safe SMILES"):
            batch, _smiles, _molIndex = batch_data
            _osmiles+=_smiles
        _odf = pd.DataFrame(_osmiles)
        _odf.to_csv(args.save_safe_smiles_file,index=False,header=None)
    #sys.exit()
    #Load model
    if len(args.model_config)>1:
        model_args = load_namespace_from_json(
            args.model_config,
            extra_args={'vocab': args.vocab, 'atom_vocab': args.atom_vocab}
        )
    else:
        model_args = args
    log_msg("Loading model with args: {}".format(model_args))
    
    model = HierAE(model_args).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.25,
        patience=args.loss_increase_tolerance,
        min_lr=1e-6)
    last_lr = optimizer.param_groups[0]['lr']
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.5) # Bryan's way of updating LR. I used this one for fine tuning. 

    if os.path.exists(args.trained_model) and os.path.isfile(args.trained_model):
        log_msg(f'Loading from checkpoint {args.trained_model}')
        #model_state, _, total_steps = torch.load(args.trained_model)
        _ = torch.load(args.trained_model)
        total_steps = _[2]
        log_msg("loaded stuff:")
        log_msg("end loaded stuff")
        #sys.exit(0)
        model_state = _[0]
        model.load_state_dict(model_state)
        optimizer_state = _[1]
        #optimizer.load_state_dict(optimizer_state)
    else:
        total_steps = 0
    #
    ref_type = type(torch.tensor([0]))
    # tolerance = args.loss_increase_tolerance # part of Bryan's way of updating LR
    # loss_increase_tolerance = tolerance # steps # part of Bryan's way of updating LR
    #
    print_inner_iter = args.print_inner_iter>0
    best_loss = args.best_loss
    save_best_loss = args.save_best_loss
    
    default_meters = np.zeros(5)
    inner_meters = default_meters

    for epoch in range(1,args.epochs+1):
        meters = default_meters
        inner_batch = 0
        # print_times = 0 # part of Bryan's way of updating LR
        #
        log_msg(f'Epoch {epoch} training...')
        for batch_data in tqdm(dataloader):
            batch, _smiles, _molIndex = batch_data
            inner_batch+=1
            total_steps+=1
            model.zero_grad()
            loss, wacc, iacc, tacc, sacc = model(*batch)
            if torch.isnan(loss ):
                log_msg("Loss is nan. Exit.")
                sys.exit(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            ms = [loss.item(),wacc,iacc,tacc,sacc]
            msf= [1,100,100,100,100]
            for _i,_m_mf in enumerate(zip(ms,msf)):
                _m,_mf = _m_mf[0],_m_mf[1]
                if type(_m)==ref_type:
                    _m = _m.cpu().numpy()
                ms[_i] = _m * _mf
            umeters = np.array(ms)
            meters = meters + umeters
            inner_meters = inner_meters + umeters
            print_this_iter = total_steps%args.print_inner_iter==0
            if print_inner_iter and print_this_iter:
                # print_times+=1 # part of Bryan's way of updating LR
                inner_meters/=args.print_inner_iter
                base_m = f"Total steps: {total_steps}. Epoch: {epoch}/{args.epochs}. Inner epoch: {inner_batch}/{total_batches}. "
                #m+=" | loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (*inner_meters, param_norm(model), grad_norm(model))
                m = base_m
                m+=f" | loss: {inner_meters[0]:.2f}, Word: {inner_meters[1]:.2f}, Assm: {inner_meters[4]:.2f}, "
                m+=f"Wo2: {inner_meters[2]:.2f}, Topo: {inner_meters[3]:.2f}, "
                m+=utils.get_cuda_usage()
                log_msg(m)
                loss = inner_meters[0]
                # if print_times==1:
                #     prev_loss = loss
                # else:
                #     if loss>prev_loss:
                #         loss_increase_tolerance-=1
                #     else:
                #         prev_loss = loss
                #         loss_increase_tolerance=min(loss_increase_tolerance+1,tolerance)
                #     if loss_increase_tolerance<=0:
                #         scheduler.step()
                #         loss_increase_tolerance=tolerance
                #         current_lr = optimizer.param_groups[0]['lr']
                #         current_lr = max(current_lr, min_lr)
                #         for param_group in optimizer.param_groups:
                #             param_group['lr'] = current_lr
                #         log_msg(f'Decreasing Learning Rate to: {current_lr}')
                scheduler.step(loss.item()) # this is my way of adapting LR. Bryan's is the commented one. For fine-tuning I used his
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                if current_lr != last_lr:
                    logging.info(f"Decreasing Learning Rate to: {current_lr:.6f}")
                    last_lr = current_lr
                inner_meters = default_meters
                if save_best_loss:
                    if loss<best_loss:
                        best_loss=loss
                        args.best_loss=loss
                        utils.save_model(model,optimizer,total_steps,time_now,args,log_msg,model_tag="best_loss")
            if (total_steps%args.save_total_steps==0) and (args.save_total_steps>0):
                utils.save_model(model,optimizer,total_steps,time_now,args,log_msg,model_tag=f"{total_steps:08d}")
            torch.cuda.empty_cache()
            del  loss, wacc, iacc, tacc, sacc
            gc.collect() #Added
            torch.cuda.empty_cache()

        meters /= len(dataset)
        if epoch%args.print_iter==0:
            m = f"Total steps: {total_steps}. Epoch: {epoch}/{args.epochs}. Inner epoch: --/--. "
            m+=f" | loss: {meters[0]:.2f}, Word: {meters[1]:.2f}, Assm: {meters[4]:.2f}, "
            m+=f"Wo2: {meters[2]:.2f}, Topo: {meters[3]:.2f}, "
            log_msg(m)

    if args.save_at_end==1:
        utils.save_model(model,optimizer,total_steps,time_now,args,log_msg,model_tag=f"{total_steps:08d}")
