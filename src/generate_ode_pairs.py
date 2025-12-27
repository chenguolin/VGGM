import os
from tqdm import tqdm
import torch
import accelerate
from accelerate import Accelerator

import sys; sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for src modules
from src.options import opt_dict, ROOT
from src.data import *  # import all dataset classes and `yield_forever`
from src.models.networks import WanVAEWrapper
from src.models import Wan
from src.utils import plucker_ray


BATCH_SIZE_PER_DEVICE = 2
NUM_ITERS = 20


@torch.autocast("cuda", dtype=torch.bfloat16)
@torch.no_grad()
def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    accelerate.utils.set_seed(0)

    os.makedirs(f"{ROOT}/data/ode_pairs_t2v", exist_ok=True)

    opt = opt_dict["wan2.1_t2v_1.3b"]
    opt.generator_path = f"{ROOT}/projects/VGGM/.pth"

    opt.input_res = (288, 512)
    opt.num_inference_steps = 48
    opt.cfg_scale = (5.,)

    accelerator = Accelerator()

    model = Wan(opt)
    model.eval()
    model.diffusion.scheduler.set_timesteps(opt.num_inference_steps, training=False)

    vae = WanVAEWrapper(opt.vae_path)
    vae.eval()

    train_dataset = RealcamvidDataset(opt, training=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_DEVICE,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
        collate_fn=BaseDataset.collate_fn,
    )

    model, vae, train_loader = accelerator.prepare(model, vae, train_loader)

    for num_iters, data in tqdm(enumerate(train_loader),
        desc="Generating ODE Pairs", ncols=125, total=NUM_ITERS):
        images = data["image"]
        B, F, _, H, W = images.shape

        idxs = torch.arange(0, F, 4).to(device=accelerator.device, dtype=torch.long)
        C2W = data["C2W"][:, idxs, ...]  # (B, f, 4, 4)
        fxfycxcy = data["fxfycxcy"][:, idxs, ...]  # (B, f, 4)
        plucker = plucker_ray(H, W, C2W.float(), fxfycxcy.float())[0].to(images.dtype)  # (B, f, 6, H, W)

        f = 1 + (F - 1) // opt.compression_ratio[0]
        h = H // opt.compression_ratio[1]
        w = W // opt.compression_ratio[2]

        prompts = data["prompt"]
        accelerator.unwrap_model(model).text_encoder.eval()
        prompt_embeds = accelerator.unwrap_model(model).text_encoder(prompts)
        negative_prompt_embeds = accelerator.unwrap_model(model).text_encoder([opt.negative_prompt]).repeat(B, 1, 1)

        if opt.first_latent_cond:
            cond_latents = accelerator.unwrap_model(model).encode(images[:, 0:1, ...] * 2. - 1., vae)
        else:
            cond_latents = None

        noisy_latents = []
        latents = torch.randn(B, opt.latent_dim, f, h, w, device="cuda")
        if cond_latents is not None:
            latents = torch.cat([cond_latents, latents[:, :, 1:, ...]], dim=2)

        for i, timestep in enumerate(accelerator.unwrap_model(model).diffusion.scheduler.timesteps):
            timesteps = timestep * torch.ones(B, f, device="cuda")
            if cond_latents is not None:
                timesteps = torch.cat([torch.zeros_like(timesteps[:, :1]), timesteps[:, 1:]], dim=1)

            noisy_latents.append(latents)

            model_outputs = accelerator.unwrap_model(model).diffusion(
                latents,
                timesteps,
                prompt_embeds,
                plucker=plucker,
            )
            model_outputs_neg = accelerator.unwrap_model(model).diffusion(
                latents,
                timesteps,
                negative_prompt_embeds,
                plucker=plucker,
            )
            model_outputs = model_outputs_neg + opt.cfg_scale[0] * (model_outputs - model_outputs_neg)

            latents = accelerator.unwrap_model(model).diffusion.scheduler.step(
                model_outputs.transpose(1, 2).flatten(0, 1),
                timesteps.flatten(0, 1),
                latents.transpose(1, 2).flatten(0, 1),
            ).unflatten(0, (B, f)).transpose(1, 2)  # (B, D, f, h, w)

        noisy_latents.append(latents)
        noisy_latents = torch.stack(noisy_latents, dim=1)  # (B, T+1, D, f, h, w)
        noisy_latents = noisy_latents[:, [[0, 12, 24, 36, -1]], ...]

        # pred_images = (accelerator.unwrap_model(model).decode_latent(latents, vae).clamp(-1., 1.) + 1.) / 2.
        # import imageio.v2 as iio
        # from src.utils.vis_util import tensor_to_video
        # iio.mimwrite("temp.mp4", tensor_to_video(pred_images))
        # exit()

        for bi in range(B):
            if opt.first_latent_cond:
                ode_pairs = {
                    "noisy_latents": noisy_latents[bi].to("cpu").squeeze(0),  # (T+1, D, f, h, w)
                    "cond_latents": cond_latents[bi].to("cpu"),  # (D, 1, h, w)
                    "prompt_embeds": prompt_embeds[bi].to("cpu"),  # (N, D)
                    "plucker": plucker[bi].to("cpu"),  # (f, 6, H, W)
                }
            else:
                ode_pairs = {
                    "noisy_latents": noisy_latents[bi].to("cpu")[0].squeeze(0),  # (T+1, D, f, h, w)
                    "prompt_embeds": prompt_embeds[bi].to("cpu"),  # (N, D)
                    "plucker": plucker[bi].to("cpu"),  # (f, 6, H, W)
                }

            idx = num_iters * B * accelerator.num_processes + accelerator.process_index * B + bi
            torch.save(ode_pairs, f"{ROOT}/data/ode_pairs_t2v/temp_{idx:06d}.pt")

        accelerator.wait_for_everyone()
        if num_iters == NUM_ITERS-1:
            break


if __name__ == "__main__":
    main()
