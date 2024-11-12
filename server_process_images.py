import time
import json
import os
import base64
import argparse
import numpy as np

import torch
from einops import rearrange
from PIL import Image

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long
from dataclasses import dataclass
from server_queue_manager import check_pending_queue, mark_request_processed
from server_email_sender import send_email_with_images
from datetime import datetime

PROCESSED_IMAGES_DIR = "/app/processed_images"
DECODED_IMAGES_DIR = "/app/decoded_images"
os.makedirs(DECODED_IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

### MODEL FUNCTIONS
def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip

class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args.fp8,
        )
        self.pulid_model = PuLIDPipeline(self.model, device="cpu" if offload else device, weight_dtype=torch.bfloat16,
                                         onnx_provider=args.onnx_provider)
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = self.pulid_model.face_helper.face_det.mean_tensor.to(torch.device("cuda"))
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args.pretrained_model, version=args.version)

    @torch.inference_mode()
    def generate_image(
            self,
            width,
            height,
            num_steps,
            start_step,
            guidance,
            seed,
            prompt,
            id_image=None,
            id_weight=1.0,
            neg_prompt="",
            true_cfg=1.0,
            timestep_to_start_cfg=1,
            max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt) if use_true_cfg else None

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(id_image, cal_uncond=use_true_cfg)
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model, **inp, timesteps=timesteps, guidance=opts.guidance, id=id_embeddings, id_weight=id_weight,
            start_step=start_step, uncond_id=uncond_id_embeddings, true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed), self.pulid_model.debug_img_list
### MODEL FUNCTIONS END
    
### HELPER FUNCTIONS

### Decode base64image
def decode_base64_image(base64_str, filename):
    """Decodes a base64 string and saves it as a .png file."""
    image_data = base64.b64decode(base64_str)
    file_path = os.path.join(DECODED_IMAGES_DIR, f"{filename}.png")
    with open(file_path, "wb") as img_file:
        img_file.write(image_data)
    return file_path

### Photostyles loading
def load_photostyles(file_path='photostyles.json'):
    with open(file_path, 'r') as f:
        return json.load(f)

# Initialize the model and return it
def initialize_model(args):
    # Initialize and return the model using the provided arguments
    flux_generator = FluxGenerator(
        model_name=args.name,
        device=args.device,
        offload=args.offload,
        aggressive_offload=args.aggressive_offload,
        args=args
    )
    return flux_generator

### GENERATION PARAMETERS (without seed, id_image, prompt and neg_prompt)
generation_parameters = {
    'width': 896,
    'height': 1152,
    'num_steps': 20,
    'start_step': 0,
    'guidance': 4,
    'id_weight': 1.05,
    'true_cfg': 1,
    'timestep_to_start_cfg': 1.0,
    'max_sequence_length': 256
}

photostyles_data = load_photostyles()

def process_and_save_images(model):
    pending_requests = check_pending_queue()
    print("Pending requests loaded:", len(pending_requests))
    
    for request in pending_requests:
        email = request['email']
        photostyle = request['photoStyle']
        photo_base64 = request['photo']
        
        style_params = photostyles_data['photostyles'].get(photostyle)
        # Decode and process image
        decoded_image_path = decode_base64_image(photo_base64, email)
        try:
            img = Image.open(decoded_image_path)
            id_image = np.array(img)
        except Exception as e:
            raise ValueError(f"Error loading image from {decoded_image_path}: {e}")
        processed_images_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output_images/{request['email']}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        for i in range(3):
            processed_image, seed, _ = model.generate_image(
                width = generation_parameters['width'],
                height = generation_parameters['height'],
                num_steps = generation_parameters['num_steps'],
                start_step = generation_parameters['start_step'],
                guidance = generation_parameters['guidance'],
                seed = -1,
                prompt = style_params['prompt'],
                id_image = id_image,
                id_weight = generation_parameters['id_weight'],
                neg_prompt = style_params['neg_prompt'],
                true_cfg = generation_parameters['true_cfg'],
                timestep_to_start_cfg = generation_parameters['timestep_to_start_cfg'],
                max_sequence_length = generation_parameters['max_sequence_length'],
    )
            processed_image_name = f"generated_{i + 1}.png"
            processed_image_path = os.path.join(output_dir, processed_image_name)
            processed_image.save(processed_image_path)
            print(f'image {processed_image_name} saved!')
            processed_images_paths.append(processed_image_path)
        
        # Mark request as processed in JSON and initiate email
        request['processed_images'] = processed_images_paths
        mark_request_processed(request)

        # Send email notification with processed images
        send_email_with_images(email, processed_images_paths)