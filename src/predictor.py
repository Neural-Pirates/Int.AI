import os
import torch
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
)
from controlnet_aux import MLSDdetector
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from colors import ade_palette
from utils import map_colors_rgb
import argparse


def filter_items(colors_list, items_list, items_to_remove):
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items


def setup_models():
    controlnet = [
        ControlNetModel.from_pretrained("BertChristiaens/controlnet-seg-room"),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd"),
    ]

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V3.0_VAE",
        controlnet=controlnet,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cpu")  # Use CPU

    seg_image_processor = AutoImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    image_segmentor = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    mlsd_processor = MLSDdetector.from_pretrained("lllyasviel/Annotators")

    return pipe, seg_image_processor, image_segmentor, mlsd_processor


def segment_image(image, seg_image_processor, image_segmentor):
    pixel_values = seg_image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = seg_image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    return Image.fromarray(color_seg).convert("RGB")


def resize_dimensions(dimensions, target_size):
    width, height = dimensions
    if width < target_size and height < target_size:
        return dimensions
    if width > height:
        aspect_ratio = height / width
        return target_size, int(target_size * aspect_ratio)
    else:
        aspect_ratio = width / height
        return int(target_size * aspect_ratio), target_size


def predict(
    pipe,
    image_path,
    prompt,
    seg_image_processor,
    image_segmentor,
    mlsd_processor,
    seed=None,
    negative_prompt="",
    output_dir=None,
):
    if seed is None:
        seed = int.from_bytes(os.urandom(2), "big")

    img = Image.open(image_path).convert("RGB")

    if "bedroom" in prompt and "bed " not in prompt:
        prompt += ", with a queen size bed against the wall"
    elif "children room" in prompt or "children's room" in prompt:
        if "bed " not in prompt:
            prompt += ", with a twin bed against the wall"

    pos_prompt = (
        prompt
        + ", interior design, 4K, high resolution, elegant, tastefully decorated, functional"
    )

    orig_w, orig_h = img.size
    new_width, new_height = resize_dimensions(img.size, 512)
    input_image = img.resize((new_width, new_height))

    # Preprocess for segmentation ControlNet
    real_seg = np.array(
        segment_image(input_image, seg_image_processor, image_segmentor)
    )
    unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_rgb(i) for i in unique_colors]
    chosen_colors, _ = filter_items(
        colors_list=unique_colors,
        items_list=segment_items,
        items_to_remove=[
            "windowpane;window",
            "column;pillar",
            "door;double;door",
        ],
    )
    mask = np.zeros_like(real_seg)
    for color in chosen_colors:
        color_matches = (real_seg == color).all(axis=2)
        mask[color_matches] = 1

    image_np = np.array(input_image)
    image = Image.fromarray(image_np).convert("RGB")
    segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

    # Preprocess for MLSD ControlNet
    mlsd_img = mlsd_processor(input_image).resize(image.size)

    # Generate output
    generated_image = pipe(
        prompt=pos_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=26,
        strength=1,
        guidance_scale=5,
        generator=[torch.Generator().manual_seed(seed)],
        image=image,
        mask_image=mask_image,
        control_image=[segmentation_cond_image, mlsd_img],
        controlnet_conditioning_scale=[0.4, 0.2],
        control_guidance_start=[0, 0.1],
        control_guidance_end=[0.5, 0.25],
    ).images[0]

    out_img = generated_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    out_path = os.path.join(output_dir, "output.png") if output_dir else "output.png"
    out_img.save(out_path)

    return out_path


if "__name__" == "__main__":
    pipe, seg_image_processor, image_segmentor, mlsd_processor = setup_models()

    # Run prediction
    output_path = predict(
        pipe,
        image_path=r"test_images\living_room_3.jpg",
        prompt="create modern designed light blue themed living room",
        negative_prompt="",
        seg_image_processor=seg_image_processor,
        image_segmentor=image_segmentor,
        mlsd_processor=mlsd_processor,
        seed=None,
    )
    print(f"Generated image saved at: {output_path}")
