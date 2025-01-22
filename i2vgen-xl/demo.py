from moviepy.editor import VideoFileClip
import numpy as np
import os
from PIL import Image
import sys
sys.path.append('..')
from pathlib import Path
import torch
import argparse
import logging
from omegaconf import OmegaConf
import json

#set cuda device
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# HF imports
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
)
from diffusers.utils import load_image, export_to_video, export_to_gif

# Project imports
from utils import (
    seed_everything,
    load_video_frames,
    convert_video_to_frames,
    load_ddim_latents_at_T,
    load_ddim_latents_at_t,
)
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
from pipelines.unet_i2vgen_xl2 import I2VGenXLUNet2
from run_group_ddim_inversion import ddim_inversion, ddim_sampling


from pnp_utils import (
    register_time,
    register_conv_injection,
    register_spatial_attention_pnp,
    register_temp_attention_pnp,
)

from run_group_pnp_edit import init_pnp

video_path = "../demo/An Old Man Doing Exercises For The Body And Mind.mp4"
output_dir = "../demo/An Old Man Doing Exercises For The Body And Mind/edited_first_frame"

###Load video ################################################################

# Set up an example inversion config file
config = {
    # General
    "seed": 8888,
    "device": "cuda:1",  # <-- change this to the GPU you want to use
    "debug": False,  # For logging DEBUG level messages otherwise INFO

    # Dir
    "data_dir": "/home/sanoojan/Video_diffusion/AnyV2V",  # <-- change this to the path of the data directory, if you cloned the repo, leave it as "..", the inversion latents will be saved in AnyV2V/inversions/
    "model_name": "i2vgen-xl",
    "exp_name": "${video_name}",
    "output_dir": "${data_dir}/inversions/${model_name}/${exp_name}",

    # Data
    "image_size": [512,512],
    "video_dir": "${data_dir}/demo",
    "video_name": "An Old Man Doing Exercises For The Body And Mind",
    # "video_name": "A guy reading the news",
    "video_frames_path": "${video_dir}/${video_name}",

    # DDIM settings
    "n_frames": 16,

    # DDIM inversion
    "inverse_config": {
        "image_size": "${image_size}",
        "n_frames": "${n_frames}",
        "cfg": 1.0,
        "target_fps": 8,
        "prompt": "A very sad and depressed guy",
        "negative_prompt": "",
        "n_steps": 500,
        "output_dir": "${output_dir}/ddim_latents",
    },

    # DDIM reconstruction
    "recon_config": {
        "enable_recon": False,
        "image_size": "${image_size}",
        "n_frames": "${n_frames}",
        "cfg": 9.0,
        "target_fps": 8,
        "prompt": "",
        "negative_prompt": "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
        "n_steps": 50,
        "ddim_init_latents_t_idx": 3,  # 0 for 981, 3 for 921, 9 for 801, 20 for 581 if n_steps=50
        "ddim_latents_path": "${inverse_config.output_dir}"
    }
}

# Convert the dictionary to an OmegaConf object
config = OmegaConf.create(config)

# Set up logging
logging_level = logging.DEBUG if config.debug else logging.INFO
logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"config: {OmegaConf.to_yaml(config)}")

# Set up device and seed
device = torch.device(config.device)
torch.set_grad_enabled(False)
seed_everything(config.seed)

logger.info(f"Loading frames from: {config.video_frames_path}")
_, frame_list = load_video_frames(config.video_frames_path, config.n_frames, config.image_size)


###DDIM inversion ################################################################

# Initialize the pipeline
pipe = I2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl",
        torch_dtype=torch.float16,
        variant="fp16",
)
custom_unet = I2VGenXLUNet2(**pipe.unet.config)
custom_unet.load_state_dict(pipe.unet.state_dict()) 
custom_unet=custom_unet.to(torch.float16)
pipe.unet = custom_unet 

device="cuda:1"
pipe.to(device)
g = torch.Generator(device=device)
g = g.manual_seed(config.seed)

# Initialize the DDIM inverse scheduler
inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        "ali-vilab/i2vgen-xl",
        subfolder="scheduler",
)
# Initialize the DDIM scheduler
ddim_scheduler = DDIMScheduler.from_pretrained(
        "ali-vilab/i2vgen-xl",
        subfolder="scheduler",
)

first_frame = frame_list[0]  # Is a PIL image

#if not config.inverse_config.output_dir is exists then run the inversion

inverse_conf_path=Path(config.inverse_config.output_dir)
if not inverse_conf_path.exists():
    # Run DDIM inversion
    _ddim_latents = ddim_inversion(config.inverse_config, first_frame, frame_list, pipe, inverse_scheduler, g)
    logger.info(f"Saved inversion latents to: {config.inverse_config.output_dir}")

# _ddim_latents = ddim_inversion(config.inverse_config, first_frame, frame_list, pipe, inverse_scheduler, g)
# Reconstruction
recon_config = config.recon_config
if recon_config.enable_recon: # Default False
            ddim_init_latents_t_idx = recon_config.ddim_init_latents_t_idx
            ddim_scheduler.set_timesteps(recon_config.n_steps)
            logger.info(f"ddim_scheduler.timesteps: {ddim_scheduler.timesteps}")
            ddim_latents_path = recon_config.ddim_latents_path
            ddim_latents_at_t = load_ddim_latents_at_t(
                ddim_scheduler.timesteps[ddim_init_latents_t_idx],
                ddim_latents_path=ddim_latents_path,
            )
            logger.debug(f"ddim_scheduler.timesteps[t_idx]: {ddim_scheduler.timesteps[ddim_init_latents_t_idx]}")
            reconstructed_video = ddim_sampling(
                recon_config,
                first_frame,
                ddim_latents_at_t,
                pipe,
                ddim_scheduler,
                ddim_init_latents_t_idx,
                g,
            )

            # Save the reconstructed video
            os.makedirs(config.output_dir, exist_ok=True)
            # Downsampling the video for space saving
            reconstructed_video = [frame.resize((512, 512), resample=Image.LANCZOS) for frame in reconstructed_video]
            export_to_video(
                reconstructed_video,
                os.path.join(config.output_dir, "ddim_reconstruction.mp4"),
                fps=10,
            )
            export_to_gif(
                reconstructed_video,
                os.path.join(config.output_dir, "ddim_reconstruction.gif"),
            )
            logger.info(f"Saved reconstructed video to {config.output_dir}")
            

##DDIM sampling and pnp injection ################################################################ 

# Set up an example sampling config file
config = {
    # General
    "seed": 8888,
    "device": "cuda:2",  # <-- change this to the GPU you want to use
    "debug": False,  # For logging DEBUG level messages otherwise INFO

    # Dir
    "data_dir": "/home/sanoojan/Video_diffusion/AnyV2V",  # <-- change this to the path of the data directory, if you cloned the repo, leave it as "..", the inversion latents will be saved in AnyV2V/
    "model_name": "i2vgen-xl",
    "task_name": "Prompt-Based-Editing",
    "edited_video_name": "Yann is doing exercises for the body and mind 10 frames_w_emb_wo_latents",
    "output_dir": "${data_dir}/Results/${task_name}/${model_name}/${video_name}/${edited_video_name}/",

    # Data
    "image_size": [512,512],
    "video_dir": "${data_dir}/demo",
    # "video_name":"A guy reading the news",
    "video_name": "An Old Man Doing Exercises For The Body And Mind",
    "video_frames_path": "${video_dir}/${video_name}",
    "edited_first_frame_path":"${data_dir}/demo/An Old Man Doing Exercises For The Body And Mind/edited_first_frame/yann_lecunn.png",
    "edited_frames_path":"${data_dir}/demo/An Old Man Doing Exercises For The Body And Mind/edited_first_frame/Yann",
    "ddim_latents_path": "${data_dir}/inversions/${model_name}/${video_name}/ddim_latents",  # Same as inverse_config.output_dir

    # Pnp Editing
    "n_frames": 16,
    "n_edited_frames": 10,
    "cfg": 9.0,
    "target_fps": 8,
    
    "editing_prompt":"a man doing exercises for the body and mind",
    "editing_negative_prompt": "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
    "n_steps": 50,
    "ddim_init_latents_t_idx": 0,  # 0 for 981, 3 for 921, 9 for 801, 20 for 581 if n_steps=50
    "ddim_inv_prompt": "",
    "random_ratio": 0.0,

    # Pnp config
    "pnp_f_t": 1.0,
    "pnp_spatial_attn_t": 1.0,
    "pnp_temp_attn_t":1.0
}

# Convert the dictionary to an OmegaConf object
config = OmegaConf.create(config)


# Set up logging
logging_level = logging.DEBUG if config.debug else logging.INFO
logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"config: {OmegaConf.to_yaml(config)}")

# Set up device and seed
device = torch.device(config.device)
torch.set_grad_enabled(False)
seed_everything(config.seed)


src_frame_list = frame_list # Loaded from step 1
src_1st_frame = src_frame_list[0]  # Is a PIL image

# Load the edited first frame
edited_1st_frame = load_image(config.edited_first_frame_path)
edited_1st_frame = edited_1st_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)

# Load the edited frames
edited_frame_list = []
num_frames_available = min(config.n_edited_frames, config.n_frames, len(os.listdir(config.edited_frames_path)))
for i in range(num_frames_available):
    edited_frame = load_image(os.path.join(config.edited_frames_path, f"{i:012d}.png"))
    edited_frame = edited_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)
    edited_frame_list.append(edited_frame)

# Load the initial latents at t
ddim_init_latents_t_idx = config.ddim_init_latents_t_idx
ddim_scheduler.set_timesteps(config.n_steps)
logger.info(f"ddim_scheduler.timesteps: {ddim_scheduler.timesteps}")
ddim_latents_at_t = load_ddim_latents_at_t(     # check this @Sanoojan
            ddim_scheduler.timesteps[ddim_init_latents_t_idx], ddim_latents_path=config.ddim_latents_path
        )
logger.debug(f"ddim_scheduler.timesteps[t_idx]: {ddim_scheduler.timesteps[ddim_init_latents_t_idx]}")
logger.debug(f"ddim_latents_at_t.shape: {ddim_latents_at_t.shape}")

# Blend the latents
random_latents = torch.randn_like(ddim_latents_at_t)
logger.info(f"Blending random_ratio (1 means random latent): {config.random_ratio}")
mixed_latents = random_latents * config.random_ratio + ddim_latents_at_t * (1 - config.random_ratio)

# Init Pnp
init_pnp(pipe, ddim_scheduler, config)

# Edit video
pipe.register_modules(scheduler=ddim_scheduler)
edited_video = pipe.sample_with_pnp(
            prompt=config.editing_prompt,
            image=edited_1st_frame,
            edited_images=edited_frame_list,
            height=config.image_size[1],
            width=config.image_size[0],
            num_frames=config.n_frames,
            num_inference_steps=config.n_steps,
            guidance_scale=config.cfg,
            negative_prompt=config.editing_negative_prompt,
            target_fps=config.target_fps,
            latents=mixed_latents,
            generator=torch.manual_seed(config.seed),
            return_dict=True,
            ddim_init_latents_t_idx=ddim_init_latents_t_idx,
            ddim_inv_latents_path=config.ddim_latents_path,
            ddim_inv_prompt=config.ddim_inv_prompt,
            ddim_inv_1st_frame=src_1st_frame,
).frames[0]

# Save video
# Add the config to the output_dir,
config_suffix = (
            "ddim_init_latents_t_idx_"
            + str(ddim_init_latents_t_idx)
            + "_nsteps_"
            + str(config.n_steps)
            + "_cfg_"
            + str(config.cfg)
            + "_pnpf"
            + str(config.pnp_f_t)
            + "_pnps"
            + str(config.pnp_spatial_attn_t)
            + "_pnpt"
            + str(config.pnp_temp_attn_t)
)
output_dir = os.path.join(config.output_dir, config_suffix)
os.makedirs(output_dir, exist_ok=True)
edited_video = [frame.resize(config.image_size, resample=Image.LANCZOS) for frame in edited_video]
edited_video_file_name = "video"
export_to_video(edited_video, os.path.join(output_dir, f"{edited_video_file_name}.mp4"), fps=config.target_fps)
export_to_gif(edited_video, os.path.join(output_dir, f"{edited_video_file_name}.gif"))
logger.info(f"Saved video to: {os.path.join(output_dir, f'{edited_video_file_name}.mp4')}")
logger.info(f"Saved gif to: {os.path.join(output_dir, f'{edited_video_file_name}.gif')}")
for i, frame in enumerate(edited_video):
      frame.save(os.path.join(output_dir, f"{edited_video_file_name}_{i:05d}.png"))
      logger.info(f"Saved frames to: {os.path.join(output_dir, f'{edited_video_file_name}_{i:05d}.png')}")