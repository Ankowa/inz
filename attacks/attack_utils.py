from typing import List, Tuple
import os

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    DiffusionPipeline,
)
from PIL import Image
from datasets import load_from_disk
from torchvision import transforms
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
import torch
import torchvision
from utils import PQDataset, load_parquet_files
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from copy import deepcopy
from accelerator import Accelerator

HG_CACHE_DIR = os.path.join(os.environ["SCRATCH"], "hg_cache")
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
get_checkpoint = lambda x, model_dir: os.path.join(model_dir, f"checkpoint-{x}")


def get_sd_from_checkpoint(args, sd_final):
    if args.checkpoint_id == -1:
        return sd_final
    elif args.checkpoint_id == 0:
        return StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            cache_dir=args.hg_cache_dir,
            torch_dtype=torch.float16,
        )
    else:
        accelerator = Accelerator()
        sd_checkpoint = deepcopy(sd_final)
        unet = accelerator.prepare(
            sd_checkpoint.unet,
        )
        print(
            "loading model from checkpoint",
            get_checkpoint(args.checkpoint_id, args.model_name),
        )
        accelerator.load_state(get_checkpoint(args.checkpoint_id, args.model_name))
    return sd_checkpoint


def get_diffusion_stuff(
    device,
    num_inference_steps,
    args,
):
    if args.checkpoint_id == -1:
        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            torch_dtype=torch.float16,
            cache_dir=HG_CACHE_DIR,
        )

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16,
            cache_dir=HG_CACHE_DIR,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16,
            cache_dir=HG_CACHE_DIR,
        )

        # 3. The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            torch_dtype=torch.float16,
            cache_dir=HG_CACHE_DIR,
        )

        # 4. Load the diffusion model and scheduler.
        vae.to(device)
        text_encoder.to(device)
        unet.to(device)

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            cache_dir=HG_CACHE_DIR,
        ).to(device)
        scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.scheduler = scheduler
        del pipeline

        scheduler.set_timesteps(num_inference_steps)

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(512),
                torchvision.transforms.CenterCrop(512),
            ]
        )

        return vae, text_encoder, unet, scheduler, tokenizer, transform

    else:
        sd_final = StableDiffusionPipeline.from_pretrained(
            args.model_name,
            local_files_only=True,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        sd_final.vae.requires_grad_(False)
        sd_final.text_encoder.requires_grad_(False)
        sd_final.unet.requires_grad_(False)

        sd = get_sd_from_checkpoint(args, sd_final)

        scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        scheduler.set_timesteps(10)

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(512),
                torchvision.transforms.CenterCrop(512),
            ]
        )

        sd.text_encoder.to(device)
        sd.unet.to(device)
        sd.vae.to(device)

        return sd.text_encoder, sd.unet, sd.vae, sd.tokenizer, scheduler, transform


def get_parquet_dataset(data_path, transform) -> PQDataset:
    """
    Get the data from the parquet file.
    """
    files = load_parquet_files(data_path)
    tables = [pq.read_table(file) for file in files]
    table = pa.concat_tables(tables)
    dataset = PQDataset(table, preprocess=transform, columns=["url", "caption", "jpg"])
    return dataset


def get_huggingface_dataset(
    data_path, transform, is_members
) -> torch.utils.data.Dataset:
    """
    Get the data from the parquet file.
    """

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["jpg"] = [transform(image) for image in images]
        examples["caption"] = [caption for caption in examples["text"]]
        return examples

    dataset = load_from_disk(data_path)
    dataset = dataset["train" if is_members else "test"].with_transform(
        preprocess_train
    )
    return dataset


def get_dataset(data_path, transform, is_parquet, is_members) -> PQDataset:
    if is_parquet:
        return get_parquet_dataset(data_path, transform)
    else:
        return get_huggingface_dataset(data_path, transform, is_members)


def collate_fn_pq(sample) -> Tuple[torch.Tensor, List[str], List[str]]:
    convert_tensor = transforms.ToTensor()
    # example[2] == is_ok flag
    try:
        imgs = list()
        captions = list()
        for example in sample:
            if example[2]:
                img = (
                    convert_tensor(example[0].resize((height, width)))
                    .unsqueeze(0)
                    .to(torch.float16)
                ).repeat(5, 1, 1, 1)
                if img.shape[1] > 3:
                    img = img[:, :3, :, :]  # remove alpha channel
                elif img.shape[1] < 3:
                    raise Exception("Image has less than 3 channels")
                captions += [example[1]["caption"]] * 5
                imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        assert imgs.shape[0] == len(captions)
        return imgs, captions
    except Exception as e:
        print("collate_fn", e)
        return None, None


def collate_fn_hg(sample) -> Tuple[torch.Tensor, List[str], List[str]]:
    convert_tensor = transforms.ToTensor()
    # example[2] == is_ok flag
    try:
        imgs = list()
        captions = list()
        for example in sample:
            img = (
                convert_tensor(example["jpg"].resize((height, width)))
                .unsqueeze(0)
                .to(torch.float16)
            ).repeat(5, 1, 1, 1)
            if img.shape[1] > 3:
                img = img[:, :3, :, :]  # remove alpha channel
            elif img.shape[1] < 3:
                raise Exception("Image has less than 3 channels")
            captions += [example["caption"]] * 5
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        assert imgs.shape[0] == len(captions)
        return imgs, captions
    except Exception as e:
        print("collate_fn", e)
        return None, None


def get_batch(
    loader: torch.utils.data.DataLoader = None,
    transform: torchvision.transforms.Compose = None,
    filepaths: list = None,
    filenames: list = None,
    generations_per_sample: int = 1,
):
    if loader is not None:
        for idx, (sample, prompt) in enumerate(loader):
            print(idx, "generator batch")
            if sample is None:
                continue
            yield sample, prompt
    elif filepaths is not None:
        for filepath, filename in zip(filepaths, filenames):
            try:
                sample = transform(Image.open(filepath).convert("RGB"))
                prompt = [filename] * generations_per_sample
            except:
                continue
            yield sample, prompt


def get_batch_generator(
    args,
    transform,
    generations_per_sample,
    is_parquet=True,
    is_members=None,
    shuffle=False,
):
    datapath = args.input_dir
    dataset = get_dataset(datapath, transform, is_parquet, is_members)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_pq if is_parquet else collate_fn_hg,
    )
    batch_generator = get_batch(
        loader=loader, generations_per_sample=generations_per_sample
    )

    print("LOADED DATA", len(dataset))
    return batch_generator


def get_text_embeddings(
    tokenizer, text_encoder, prompts, do_classifier_free_guidance, device
):
    text_input = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    if do_classifier_free_guidance:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings


@torch.no_grad()
def get_image_pred(vae, latents):
    scaled_latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(scaled_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    return images


@torch.no_grad()
def get_single_timestep_run(
    scheduler,
    t,
    latents,
    text_embeddings,
    unet,
    do_classifier_free_guidance,
    num_inference_steps,
    guidance_scale,
):
    timesteps = scheduler.timesteps

    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order

    # expand the latents if we are doing classifier free guidance
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
    # predict the noise residual
    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=text_embeddings,
    ).sample

    # perform guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    if do_classifier_free_guidance:
        return latents, noise_pred_text
    return latents, noise_pred


def get_l2_loss(true, pred, is_batch=True):
    loss = np.mean(
        np.square(true.detach().cpu() - pred.detach().cpu()).numpy(),
        axis=(1, 2, 3),
    )
    return get_loss_shape(loss, is_batch)


def get_image_loss(true, pred, is_batch=True):
    loss = np.mean(
        np.abs(255 * true.cpu().permute(0, 2, 3, 1).numpy() - pred),
        axis=(1, 2, 3),
    )
    return get_loss_shape(loss, is_batch)


def get_loss_shape(loss, is_batch):
    if is_batch:
        return loss.reshape(-1, 1, 5)
    return loss.reshape(-1, 5)


def save(
    args,
    results_latent_loss,
    results_image_loss,
    results_noise_loss,
    results_prompts,
    is_panic=False,
):
    np.savez(
        os.path.join(
            args.output_dir,
            f"{'panic' if is_panic else ''}_{args.task_id if args.parallel_tasks_cnt > 1 else ''}_{args.output_filename}",
        ),
        latent_loss=np.concatenate(results_latent_loss, axis=0),
        image_loss=np.concatenate(results_image_loss, axis=0),
        noise_loss=np.concatenate(results_noise_loss, axis=0),
        prompts=results_prompts,
    )


def get_pokemon_diffusion(
    args,
    device,
    num_inference_steps,
):
    sd = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        cache_dir=HG_CACHE_DIR,
        local_files_only=True,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    sd.vae.requires_grad_(False)
    sd.text_encoder.requires_grad_(False)
    sd.unet.requires_grad_(False)
    sd.to(device)

    sd.scheduler.set_timesteps(num_inference_steps)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(512),
            torchvision.transforms.CenterCrop(512),
        ]
    )

    return sd.vae, sd.text_encoder, sd.unet, sd.scheduler, sd.tokenizer, transform


def get_starting_latent(scheduler, true_latents, noise, starting_timestep):
    sqrt_alpha_prod = scheduler.alphas_cumprod[starting_timestep] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - scheduler.alphas_cumprod[starting_timestep]) ** 0.5
    return sqrt_alpha_prod * true_latents + sqrt_one_minus_alpha_prod * noise


def get_scheduler_timesteps(
    scheduler, starting_timestep, end_timestep, num_inference_steps
):
    assert not num_inference_steps % 10, "num_inference_steps must be divisive by 10"
    multi = 1000 / num_inference_steps
    lb = -int((starting_timestep - 1) // multi + 1)
    if end_timestep == 1:
        return scheduler.timesteps[lb:]
    rb = -int((end_timestep - 1) // multi)
    return scheduler.timesteps[lb:rb]
