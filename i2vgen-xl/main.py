import os
import torch
import logging
from pathlib import Path
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip
from PIL import Image
from diffusers import DDIMInverseScheduler, DDIMScheduler
from diffusers.utils import load_image, export_to_video, export_to_gif

# Project imports
from utils import (
    seed_everything,
    load_video_frames,
    load_ddim_latents_at_t,
)
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
from pipelines.unet_i2vgen_xl2 import I2VGenXLUNet2
from run_group_ddim_inversion import ddim_inversion, ddim_sampling
from pnp_utils import register_time, register_conv_injection
from run_group_pnp_edit import init_pnp
# set cuda visible device 2

Precision=torch.float16


def setup_logging(debug):
    logging_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def main(config_path):
    # Load config
    config = OmegaConf.load(config_path)

    # Set up logging
    logger = setup_logging(config.debug)
    logger.info(f"Loaded config: {OmegaConf.to_yaml(config)}")

    # Set up device and seed
    device = torch.device(config.device)
    torch.set_grad_enabled(False)
    seed_everything(config.seed)

    # Load video frames
    logger.info(f"Loading frames from: {config.video_frames_path}")
    _, frame_list = load_video_frames(config.video_frames_path, config.n_frames, config.image_size,naming_scheme=config.naming_scheme)
    first_frame = frame_list[0]

    # Initialize pipeline
    pipe = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=Precision, variant="fp16")
    
    # Handling multiple edited frames
    
    custom_unet = I2VGenXLUNet2(**pipe.unet.config)
    custom_unet.load_state_dict(pipe.unet.state_dict())
    custom_unet = custom_unet.to(Precision)
    pipe.unet = custom_unet
    
    pipe.to(device)
    g = torch.Generator(device=device).manual_seed(config.seed)

    # Initialize schedulers
    inverse_scheduler = DDIMInverseScheduler.from_pretrained("ali-vilab/i2vgen-xl", subfolder="scheduler")
    ddim_scheduler = DDIMScheduler.from_pretrained("ali-vilab/i2vgen-xl", subfolder="scheduler")
    edited_1st_frame = load_image(config.edited_first_frame_path).resize(config.image_size, resample=Image.LANCZOS)
    # Perform DDIM inversion
    inverse_conf_path = Path(config.inverse_config.output_dir)
    if not inverse_conf_path.exists() or config.inverse_config.force_inversion:
        if config.null_optimization.null_optimization:
            _ddim_latents = ddim_inversion(config.inverse_config, first_frame, frame_list, pipe, inverse_scheduler, g)  # [n_steps, 4,n_frames,,64,64]
            _null_latents=pipe.null_optimization(_ddim_latents, 
                                                         config.null_optimization,
                                                         first_frame,
                                                         ddim_inv_prompt=config.inverse_config.prompt,edited_1st_frame=edited_1st_frame)
            
        elif config.pnp_inversion.pnp_inversion:
            _ddim_latents = ddim_inversion(config.inverse_config, first_frame, frame_list, pipe, inverse_scheduler, g)
            _pnp_latents=pipe.pnp_inversion(_ddim_latents,
                                            config.pnp_inversion,
                                            first_frame,
                                            edited_1st_frame=edited_1st_frame)
            
            # find uncond inversion latents (null_inversion_embedded)

        else:
            _ddim_latents = ddim_inversion(config.inverse_config, first_frame, frame_list, pipe, inverse_scheduler, g)
        logger.info(f"Saved inversion latents to: {config.inverse_config.output_dir}")

    # Perform DDIM reconstruction (if enabled)
    if config.recon_config.enable_recon:
        ddim_scheduler.set_timesteps(config.recon_config.n_steps)
        ddim_latents_at_t = load_ddim_latents_at_t(
            ddim_scheduler.timesteps[config.recon_config.ddim_init_latents_t_idx], config.ddim_latents_path
        )
        reconstructed_video = ddim_sampling(
            config.recon_config, first_frame, ddim_latents_at_t, pipe, ddim_scheduler, config.recon_config.ddim_init_latents_t_idx, g
        )

        # Save reconstructed video
        os.makedirs(config.output_dir, exist_ok=True)
        reconstructed_video = [frame.resize((512, 512), resample=Image.LANCZOS) for frame in reconstructed_video]
        export_to_video(reconstructed_video, os.path.join(config.output_dir, "ddim_reconstruction.mp4"), fps=10)
        export_to_gif(reconstructed_video, os.path.join(config.output_dir, "ddim_reconstruction.gif"))
        logger.info(f"Saved reconstructed video to {config.output_dir}")

    
    
    src_frame_list = frame_list # Loaded from step 1
    src_1st_frame = src_frame_list[0]  # Is a PIL image 
    
    # Load edited first frame and frames
    edited_1st_frame = load_image(config.edited_first_frame_path).resize(config.image_size, resample=Image.LANCZOS)
    edited_frames = [
        load_image(os.path.join(config.edited_frames_path, f"{i:0{config.edited_scheme}d}.png")).resize(config.image_size, resample=Image.LANCZOS)
        for i in range(min(config.editing.n_edited_frames, config.n_frames, len(os.listdir(config.edited_frames_path))))
    ]

    # Load latents
    ddim_scheduler.set_timesteps(config.editing.n_steps)
    ddim_latents_at_t = load_ddim_latents_at_t(
        ddim_scheduler.timesteps[config.editing.ddim_init_latents_t_idx], config.ddim_latents_path
    )

    # Blend latents
    random_latents = torch.randn_like(ddim_latents_at_t)
    mixed_latents = random_latents * config.editing.random_ratio + ddim_latents_at_t * (1 - config.editing.random_ratio)

    # Initialize PnP
    init_pnp(pipe, ddim_scheduler, config.pnp)

    # Edit video
    pipe.register_modules(scheduler=ddim_scheduler)
    edited_video,original_video = pipe.sample_with_pnp(
        prompt=config.editing.editing_prompt,
        image=edited_1st_frame,
        edited_images=edited_frames,
        height=config.image_size[1],
        width=config.image_size[0],
        num_frames=config.n_frames,
        num_inference_steps=config.editing.n_steps,
        guidance_scale=config.editing.cfg,
        negative_prompt=config.editing.editing_negative_prompt,
        null_latents_path=config.null_optimization.output_dir,
        null_optimization=config.null_optimization.null_optimization,
        target_fps=config.target_fps,
        latents=mixed_latents,
        generator=g,
        return_dict=True,
        ddim_init_latents_t_idx=config.editing.ddim_init_latents_t_idx,
        ddim_inv_latents_path=config.ddim_latents_path,
        ddim_inv_prompt=config.editing.ddim_inv_prompt,
        ddim_inv_1st_frame=src_1st_frame,
    )
    edited_video=edited_video.frames[0]
    original_video=original_video.frames[0]
    
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    
    # Save edited video
    output_path = os.path.join(config.output_dir, "edited_video.mp4")
    export_to_video(edited_video, output_path, fps=config.target_fps)
    logger.info(f"Saved edited video to: {output_path}")
    export_to_gif(edited_video, os.path.join(config.output_dir, "edited_video.gif"))
    logger.info(f"Saved edited video to: {output_path}")
    edited_video_file_name = "video"
    for i, frame in enumerate(edited_video):
        
      frame.save(os.path.join(config.output_dir, f"{edited_video_file_name}_{i:05d}.png"))
      logger.info(f"Saved frames to: {os.path.join(config.output_dir, f'{edited_video_file_name}_{i:05d}.png')}")
      
      
    # Save original video
    output_path = os.path.join(config.output_dir, "original_video.mp4")
    export_to_video(original_video, output_path, fps=config.target_fps)
    logger.info(f"Saved original video to: {output_path}")
    export_to_gif(original_video, os.path.join(config.output_dir, "original_video.gif"))
    logger.info(f"Saved original video to: {output_path}")
    original_video_file_name = "Original_video"
    for i, frame in enumerate(original_video):
        
      frame.save(os.path.join(config.output_dir, f"{original_video_file_name}_{i:05d}.png"))
      logger.info(f"Saved frames to: {os.path.join(config.output_dir, f'{original_video_file_name}_{i:05d}.png')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/Single_videos/CelebVHQ_video_test.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)