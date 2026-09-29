import sys
import torch # type: ignore

# Enable fast training settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from collections import OrderedDict
from torch.utils.data import DataLoader # type: ignore
from copy import deepcopy
from glob import glob
from time import time
import numpy as np
import argparse
import logging
import os, math
import torch.nn.functional as F
from tqdm import tqdm # type: ignore
from download import find_model

from models import DiT_models
from diffusion import create_diffusion

from dataset import ldmlLatentAndSignature
import random

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)

def main(args):
    """
    Trains a new DiT model on a single GPU.
    """
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)
    
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.model.replace('/', '-')}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    latent_size = 127
    in_channels = 64
    cross_attn = 768
    condition_dim = 1024 if args.text_encoder_name == 'molt5' else 4096

    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        cross_attn=cross_attn,
        condition_dim=condition_dim
    ).to(device)
    
    if args.ckpt:
        state_dict = find_model(args.ckpt)
        model.load_state_dict(state_dict, strict=True)
        logger.info(f'Loaded DiT from {args.ckpt}')
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    latents = np.load(args.latent_file)
    guidance = np.load(args.guidance)
    #Change the shape of latent if required
    if latents.shape[1:]!=(latent_size,in_channels):
        correct_shape = [latent_size,in_channels]
        m = f"Changing latent size from: {latents.shape} to : {correct_shape}"
        print(m)
        logger.info(m)
        total_shape = latent_size*in_channels
        if latents.shape[-1]<total_shape:
            latents = np.pad(latents,((0,0),(0,total_shape-latents.shape[-1])))
        elif latents.shape[-1]>total_shape:
            latents = latents[:,:total_shape]
        m = "Reshaping"
        print(m)
        logger.info(m)
        latents = np.reshape(latents,(latents.shape[0],latent_size,in_channels))
        m = "Reshaping Complete!"
        print(m)
        logger.info(m)
    dataset = ldmlLatentAndSignature(latents, guidance, device=device, max_items=args.nsample)
    print(latents.shape,guidance.shape)
    #sys.exit(0)
    
    #loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, pin_memory=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, drop_last=True)
    #loader = DataLoader(dataset, batch_size=args.global_batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    logger.info(f"Dataset contains {len(dataset):,} samples, Loader batches: {len(loader)}")

    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()
    ema.eval()  # EMA model should always be in eval mode
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    log_number = 0
    log_loss = 0
    start_time = time()
    
    best_train_loss = float('inf')
    logger.info(f"Training for {args.epochs} epochs...")
    
    best_log_loss = 1e10
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        running_loss = 0
        for x, y in tqdm(loader):
            logger.info(f"X {x.shape} Y_ {y.shape}")
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = x.permute((0, 2, 1)).unsqueeze(-1)
                # Create attention mask to only check the signature
                ys = y.shape
                y_last_dim = ys[-1]
                description_length = math.ceil(ys[1]/args.embedding_size)
                if y_last_dim>args.embedding_size:
                    # Bryan's way
                    # #Put it in the description dimension
                    # ones = torch.ones(y_last_dim).to(y.device) # the first element of the sequence is the signature, so only pay attention to it
                    # zeros = torch.zeros(description_length - y_last_dim).to(y.device)
                    # pad_mask = torch.cat([ones, zeros]).unsqueeze(0)
                    # y = y.unsqueeze(2) #batch*signature_size*1
                    # padding = torch.zeros(y.shape[0], description_length, args.embedding_size - 1).to(y.device)
                    # y = torch.cat((y, padding), dim=-1)
                    # empty_signature = torch.zeros(1,description_length,y.shape[-1]).to(y.device) # for conditional guidance

                    # My way
                    pad_mask = torch.ones(description_length).unsqueeze(0).to(y.device) # pay attention to all the elements of the sequence (3 elements if y=3052)
                    padding_needed = description_length*args.embedding_size - y_last_dim
                    y_padded = F.pad(y, (0, padding_needed)) # padd y with 0s at the end for the remaining elements
                    y = y_padded.view(args.global_batch_size, description_length, args.embedding_size).to(device)
                    empty_signature = torch.zeros(1,description_length,y.shape[-1]).to(y.device) # for conditional guidance
                    logger.info(f"y reshaped to {y.shape}. Empty sign shape {empty_signature.shape}, Pad_mask {pad_mask.shape}")

                else:
                    ones = torch.ones(1).to(y.device) # the first element of the sequence is the signature, so only pay attention to it
                    zeros = torch.zeros(description_length - 1).to(y.device)
                    pad_mask = torch.cat([ones, zeros]).unsqueeze(0)
                    #transform y
                    y = y.unsqueeze(1) #batch*1*signature_size
                    padding = torch.zeros(y.shape[0], 1, args.embedding_size - y.shape[-1]).to(y.device)
                    y = torch.cat((y, padding), dim=-1)
                    empty_signature = torch.zeros(1,1,y.shape[-1]).to(y.device) # for conditional guidance
                    logger.info(f"y reshaped to {y.shape}. Empty sign shape {empty_signature.shape}, Pad_mask {pad_mask.shape}")
                #y = [d if random.random() < 0.95 else dataset.null_text for d in y]
                #logger.info(f"{y.shape}. {empty_signature.shape}")
                #y = torch.cat([d if random.random() < 0.95 else empty_signature for d in y],dim=0)
                empty_index = [_i for _i in range(y.shape[0]) if random.random() > 0.95 ]
                y[empty_index]=empty_signature
                #y = y.to(device)  # Original: batch*len*768, Signature: batch*len*Unnkwon
                
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            #logging.info(f"x: {x.shape}, y: {y.shape}, t: {t.shape}, pad {pad_mask.shape}")
            model_kwargs = dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)
            # Log loss values:
            running_loss += loss.item()
            log_loss+=loss.item()
            log_steps += 1
            train_steps += 1
            log_number += 1
            if train_steps % args.log_every == 0:
                log_avg_loss = log_loss/log_number 
                log_loss = 0
                log_number = 0
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {log_avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if log_avg_loss < best_log_loss:
                    best_log_loss = log_avg_loss
                    checkpoint = {
                        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        "opt": opt.state_dict(),
                        "ema": ema.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/train_log_best.pt"
                    torch.save(checkpoint, checkpoint_path)
                    _m=f"Saved checkpoint to {checkpoint_path}"
                    logger.info(_m)
                    print(_m)
        
        avg_loss = running_loss / len(loader)
        logger.info(f"Epoch {epoch}: Train Loss: {avg_loss:.4f}")
        
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            checkpoint = {"model": model.state_dict(), "opt": opt.state_dict(), "ema": ema.state_dict(),"args": args}
            fo = f"{checkpoint_dir}/best_epoch_model.pt"
            # torch.save(checkpoint, fo)
            logger.info(f"Saved best model checkpoint. {fo}")
    
    logger.info("Training Complete!")
    #Save last 
    checkpoint = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "opt": opt.state_dict(),
        "ema": ema.state_dict(),
        "args": args
    }
    checkpoint_path = f"{checkpoint_dir}/train_log_last.pt"
    # torch.save(checkpoint, checkpoint_path)
    _m=f"Saved checkpoint to {checkpoint_path}"
    logger.info(_m)
    print(_m)
    #Save last 

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--text_encoder_name", type=str, default="molt5")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    # parser.add_argument("--description_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=16*6)
    parser.add_argument("--global_seed", type=int, default=42)
    
    parser.add_argument("--num_classes", type=int, default=1000)
    #parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    #
    parser.add_argument("--latent_file", type=str, required=True)
    parser.add_argument("--guidance", type=str, required=True)
    parser.add_argument("--nsample", type=int, default=-1)
    parser.add_argument("--embedding_size", type=int, default=1024)
    parser.add_argument("--working_directory", type=str, default="/aloy/home/bsaldivar/hvae_ldmol/ldmol/")  # Choice doesn't affect training
    #
    args = parser.parse_args()
    os.chdir(args.working_directory)
    print(args)
    main(args)
