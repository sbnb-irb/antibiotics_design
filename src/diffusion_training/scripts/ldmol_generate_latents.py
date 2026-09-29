import math
import os
import sys
import numpy as np
from tqdm import tqdm #type: ignore
repo_path = '/src/hvae_training'
sys.path.append(repo_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only 1 gpu
import torch.nn.functional as F
 
import torch #type: ignore
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
import argparse
from utils import check_save_directory, create_latent_dataloader, zero_pad_template
import time, logging

logging.basicConfig(level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')


@torch.no_grad()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")  # Only 1 gpu
    torch.cuda.set_device(device) # Only 1 gpu

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    # Load model:
    latent_size = 127
    in_channels = 64
    cross_attn = 768
    condition_dim = 1024
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        cross_attn=cross_attn,
        condition_dim=condition_dim,
    ).to(device)
    # # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    # ckpt_path = args.ckpt
    # state_dict = find_model(ckpt_path)
    # #msg = model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict, strict=False)

    logging.info(f'Loading from checkpoint {args.ckpt}')
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale != 1.0
    st = time.time()
    if not os.path.isfile(args.guidance):
        logging.info(f"Latent file {args.guidance} does not exist. Exit.")
        sys.exit(1)
    guidance = np.load(args.guidance)
    logging.info(f"Loaded signature. {guidance.shape}")
    n = args.batch_size
    _data_loader = create_latent_dataloader(guidance,batch_size=args.batch_size,max_items=args.nsample,device=device)
    logging.info(f"DataLoader batches. {len(_data_loader)}")
    latent_expanded_size = model.in_channels*latent_size
    output_part_template = zero_pad_template(args.samples_per_guidance)
    all_latents = []
    with torch.no_grad():
        for sample_repetition in range(args.sample_start_index,args.sample_start_index+args.samples_per_guidance):
            output_file = args.output_latents_file
            if args.samples_per_guidance > 1:
                suffix = output_part_template.format(sample_repetition)
                output_file = output_file.replace(".npy", f"_sample{suffix}.npy")
            check_save_directory(output_file)
            if guidance.shape[0]>args.batch_size and args.save_batches==1:
                batch_dir = output_file.replace(".npy","_batches/")
                check_save_directory(batch_dir)
            print(f"Sample repetition: {sample_repetition+1}/{args.samples_per_guidance}")
            _batchi = 0
            latents = []
            for y in tqdm(_data_loader):
                if args.artificial_latent_size>0:
                    z = torch.randn(y.shape[0], args.artificial_latent_size, device=device)
                    z_pad = torch.zeros(y.shape[0],latent_expanded_size - args.artificial_latent_size).to(y.device)
                    z = torch.cat([z, z_pad],dim=-1).view(y.shape[0],model.in_channels, latent_size, 1)
                else:
                    z = torch.randn(y.shape[0], model.in_channels, latent_size, 1, device=device)
                #
                ys = y.shape
                y_last_dim = ys[-1]
                print(f"y shape: {ys}")
                description_length = math.ceil(y_last_dim/args.embedding_size)
                if y_last_dim>args.embedding_size:
                    # Bryan's way
                    # #Put it in the description dimension
                    # ones = torch.ones(y_last_dim).to(y.device) # the first element of the sequence is the signature, so only pay attention to it
                    # zeros = torch.zeros(description_length - y_last_dim).to(y.device)
                    # pad_mask = torch.cat([ones, zeros]).unsqueeze(0)
                    # y = y.unsqueeze(2) #batch*signature_size*1
                    # padding = torch.zeros(y.shape[0], description_length, args.embedding_size - 1).to(y.device)
                    # y = torch.cat((y, padding), dim=-1)
                    # #empty_guidance = torch.zeros(1,args.description_length,y.shape[-1]).to(y.device) # for conditional guidance

                    # My way
                    pad_mask = torch.ones(description_length).unsqueeze(0).to(y.device) # pay attention to all the elements of the sequence (3 elements if y=3052)
                    padding_needed = description_length*args.embedding_size - y_last_dim
                    y_padded = F.pad(y, (0, padding_needed)) # padd y with 0s at the end for the remaining elements
                    if len(_data_loader)==1:
                        y = y_padded.view(len(y), description_length, args.embedding_size).to(device)
                    else:
                        y = y_padded.view(args.batch_size, description_length, args.embedding_size).to(device)
                    # empty_signature = torch.zeros(1,description_length,y.shape[-1]).to(y.device) # for conditional guidance
                    
                else:
                    ones = torch.ones(1).to(y.device) # the first element of the sequence is the signature, so only pay attention to it
                    zeros = torch.zeros(description_length - 1).to(y.device)
                    pad_mask = torch.cat([ones, zeros]).unsqueeze(0)
                    #transform y
                    y = y.unsqueeze(1) #batch*1*signature_size
                    padding = torch.zeros(y.shape[0], 1, args.embedding_size - y.shape[-1]).to(y.device)
                    y = torch.cat((y, padding), dim=-1)
                    #empty_guidance = torch.zeros(1,1,y.shape[-1]).to(y.device) # for conditional guidance
                #
                #Added:
                pad_mask = pad_mask.repeat(y.shape[0], 1)
                #
                pad_mask_cond = pad_mask
                pad_mask_null = pad_mask_cond.clone()
                #transform y
                """
                y = y.unsqueeze(1) #batch*1*signature_size
                padding = torch.zeros(y.shape[0], 1, args.embedding_size - y.shape[-1]).to(y.device)
                seq_padding = torch.zeros(y.shape[0], args.description_length - 1 , args.embedding_size).to(y.device)
                y = torch.cat((y, padding), dim=-1)
                y_cond = torch.cat((y, seq_padding), dim=1)
                """
                y_cond = y
                #y_cond = torch.cat((y, padding), dim=-1)
                y_null = torch.zeros_like(y_cond).to(y.device) # for conditional guidance
                # Setup classifier-free guidance:
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y = torch.cat([y_cond, y_null], 0)
                    pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
                    model_kwargs = dict(y=y, pad_mask=pad_mask, cfg_scale=args.cfg_scale)
                    sample_fn = model.forward_with_cfg
                else:
                    model_kwargs = dict(y=y_cond, pad_mask=pad_mask)
                    sample_fn = model.forward

                # Sample images:
                #print("\nz.shape,y.shape,pad_mask.shape,y_cond.shape,y_null.shape,pad_mask_cond.shape,pad_mask_null.shape")
                #print(z.shape,y.shape,pad_mask.shape,y_cond.shape,y_null.shape,pad_mask_cond.shape,pad_mask_null.shape)
                samples = diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                # print('zzzz', samples.shape)

                samples = samples.squeeze(-1).permute((0, 2, 1)) # reshape to original latent shape
                if args.artificial_latent_size>0:
                    #print(samples.shape)
                    samples = samples.reshape(samples.shape[0],latent_expanded_size)
                    samples = samples[:,:args.artificial_latent_size]
                samples = samples.cpu().numpy()
                if guidance.shape[0]>args.batch_size and args.save_batches==1:
                    _fout=os.path.join(batch_dir,f"batch_{_batchi:06}.npy")
                    np.save(_fout,samples)
                latents.append(samples)
                _batchi+=1
            if len(latents)>1:
                latents = np.vstack(latents)
            else:
                latents = latents[0]
            #check_save_directory(args.output_latents_file)
            # print(f"Saving:{output_file}. Size: {latents.shape}")
            # np.save(output_file,latents)
            if args.samples_per_guidance > 1:
                all_latents.append(latents)

    # After the sampling loop, stack and save all latents
    if args.samples_per_guidance > 1:
        all_latents = np.vstack(all_latents)
        output_file = args.output_latents_file
        # output_file = output_file.replace(".npy", f"_all_latents.npy")
        np.save(output_file, all_latents)
    
    logging.info(f"Saved all latents to {output_file}. Shape: {all_latents.shape}")
    logging.info("Min, max, mean and std of values in the generated latents:")
    logging.info(f"{np.min(latents)}, {np.max(latents)}, {np.mean(latents)}, {np.std(latents)}")
    logging.info(f"time: {time.time() - st}")


if __name__ == "__main__":
    logging.info("Starting latent generation script...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--guidance", type=str, default="")
    parser.add_argument("--nsample", type=int, default=-1)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--cfg_scale",  type=float, default=5.)
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    # parser.add_argument("--description_length", type=int, default=200)
    parser.add_argument("--embedding_size", type=int, default=1024)  # or any default value you prefer
    parser.add_argument("--output_latents_file", type=str, default="latents.npy")
    parser.add_argument("--artificial_latent_size", type=int, default=0)
    parser.add_argument("--working_directory", type=str, default="/aloy/home/bsaldivar/ldmol/ldmol/")  # Choice doesn't affect training
    parser.add_argument("--samples_per_guidance", type=int, default=1)
    parser.add_argument("--sample_start_index", type=int, default=0)
    parser.add_argument("--save_batches", type=int, default=0)
    args = parser.parse_args()
    os.chdir(args.working_directory)
    main(args)

