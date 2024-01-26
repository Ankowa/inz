import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from attack_utils import (
    get_diffusion_stuff,
    get_batch_generator,
    get_text_embeddings,
    get_image_pred,
    get_single_timestep_run,
    get_l2_loss,
    get_image_loss,
    save,
    get_scheduler_timesteps,
    get_starting_latent,
    get_pokemon_diffusion,
)
from attacks_config import ATTACKS

torch_device = "cuda"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="pokemon-split",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sd-pokemon-model-full-run-with-split-final2023-02-13T12:53:20.087717",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--checkpoint_id", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="pokemons")
    parser.add_argument("--output_filename", type=str, default="members_final.npz")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--parallel_tasks_cnt", type=int, default=1)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--attack", type=str, default="embeddings")

    args = parser.parse_args()
    args.input_dir = os.path.join("data/mia_in", args.input_dir)
    args.output_dir = os.path.join("data/mia_out", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    return args


@torch.no_grad()
def run(
    args,
    batch_generator,
    vae,
    text_encoder,
    unet,
    scheduler,
    tokenizer,
    device=torch_device,
):
    attack = ATTACKS[args.attack]
    results_latent_loss = []
    results_image_loss = []
    results_noise_loss = []
    results_prompts = []
    panic = False
    num_inference_steps = 1000 // attack["step"]

    timesteps_list = get_scheduler_timesteps(
        scheduler,
        starting_timestep=args["start"],
        end_timestep=args["stop"],
        num_inference_steps=num_inference_steps,
    )
    print(f"timesteps_list: {timesteps_list}")
    if attack["reverse-noise"]:
        timesteps_list = reversed(timesteps_list)
    try:
        for i in tqdm(
            range(args.num_images // (args.batch_size)),
            desc=f"task_id: {args.task_id}, {args.input_dir.split('/')[-1]}",
            total=args.num_images // (args.batch_size),
        ):
            true_images, prompts = next(batch_generator)
            if args.parallel_tasks_cnt > 1:
                if i % args.parallel_tasks_cnt != args.task_id:
                    continue

            results_prompts += prompts

            true_images = true_images.to(device)

            true_latents = vae.encode(2 * (true_images - 0.5)).latent_dist.sample()
            true_latents = true_latents * vae.config.scaling_factor

            text_embeddings = get_text_embeddings(
                tokenizer,
                text_encoder,
                prompts,
                attack["classifier-free-guidance"] != 0,
                device,
            )
            if attack["shift-text"]:
                text_embeddings += (
                    torch.randn_like(text_embeddings) * attack["shift-by"]
                )

            noise = torch.randn_like(true_latents)
            true_noise = noise

            results_batch_latent_loss = []
            results_batch_noise_loss = []
            results_batch_image_loss = []

            latents = get_starting_latent(
                scheduler,
                true_latents,
                noise,
                starting_timestep=attack["start-noise-step"],
            ).to(device)
            for idx, t in enumerate(timesteps_list):
                if idx == attack["shift-latent-idx"]:
                    shift_noise = torch.randn_like(true_latents)
                    latents += shift_noise * scheduler.init_noise_sigma

                if not attack["iterative"]:
                    latents = get_starting_latent(
                        scheduler,
                        true_latents,
                        noise,
                        starting_timestep=t
                        if not attack["fix-noise-step"]
                        else attack["start-noise-step"],
                    ).to(device)

                latents = latents * scheduler.init_noise_sigma
                latents = latents.half()

                latents, noise_pred_text = get_single_timestep_run(
                    scheduler,
                    t,
                    latents,
                    text_embeddings,
                    unet,
                    attack["classifier-free-guidance"] != 0,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=attack["classifier-free-guidance"],
                )
                loss_noise = get_l2_loss(true_noise, noise_pred_text, is_batch=True)
                loss_latent = get_l2_loss(latents, true_latents, is_batch=True)

                images = get_image_pred(vae, latents)
                loss_image = get_image_loss(true_images, images, is_batch=True)

                results_batch_latent_loss.append(loss_latent)
                results_batch_noise_loss.append(loss_noise)
                results_batch_image_loss.append(loss_image)

            results_latent_loss.append(
                np.concatenate(results_batch_latent_loss, axis=1)
            )
            results_noise_loss.append(np.concatenate(results_batch_noise_loss, axis=1))
            results_image_loss.append(np.concatenate(results_batch_image_loss, axis=1))
            torch.cuda.empty_cache()
    except Exception as e:
        print(e)
        print("PANIC")
        panic = True

    save(
        args,
        results_latent_loss,
        results_image_loss,
        results_noise_loss,
        results_prompts,
        is_panic=panic,
    )


def main():
    args = parse_args()
    print(args)
    if args.model_dir is None:
        vae, text_encoder, unet, scheduler, tokenizer, transform = get_diffusion_stuff(
            device=torch_device,
            num_inference_steps=1000 // ATTACKS[args.attack]["step"],
            args=args,
        )
        print("original SD")
    else:
        (
            vae,
            text_encoder,
            unet,
            scheduler,
            tokenizer,
            transform,
        ) = get_pokemon_diffusion(
            args,
            device=torch_device,
        )
        print("pokemon finetuned sd")
    batch_generator = get_batch_generator(
        args,
        transform,
        is_parquet=args.model_dir is None,
        is_members="nonmembers" not in args.output_filename,
    )
    run(
        args,
        batch_generator,
        vae,
        text_encoder,
        unet,
        scheduler,
        tokenizer,
        device=torch_device,
        num_inference_steps=1000 // ATTACKS[args.attack]["step"],
    )


if __name__ == "__main__":
    main()
