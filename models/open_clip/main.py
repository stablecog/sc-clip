from PIL import Image

from models.constants import DEVICE
from .constants import OPEN_CLIP_TOKEN_LENGTH_MAX
from typing import List
import torch
from utils.helpers import time_log
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
)
from concurrent.futures import ThreadPoolExecutor, as_completed


CLIP_IMAGE_SIZE = 224


def convert_to_rgb(img: Image.Image):
    return img.convert("RGB")


def create_clip_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            convert_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


clip_transform = create_clip_transform(CLIP_IMAGE_SIZE)


def clip_preprocessor(images: List[Image.Image]):
    def process_image(img: Image.Image, index: int):
        return clip_transform(img), index

    images_with_index = [(img, i) for i, img in enumerate(images)]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, img_tuple[0], img_tuple[1])
            for img_tuple in images_with_index
        ]
        results = [future.result() for future in as_completed(futures)]

    results_sorted = sorted(results, key=lambda x: x[1])
    results_sorted = [result[0] for result in results_sorted]

    return torch.stack(results_sorted)


def embeds_of_images(images: List[Image.Image], model):
    with time_log(f"[] OpenCLIP: Embedded {len(images)} image(s)"):
        with torch.no_grad():
            inputs = clip_preprocessor(images=images)
            inputs = inputs.to(DEVICE)
            vision_output = model.vision_model(pixel_values=inputs)
            image_embedding_tensors = model.visual_projection(vision_output[1])
            image_embeddings = image_embedding_tensors.cpu().numpy().tolist()
            return image_embeddings, vision_output


def embeds_of_texts(texts: str, model, tokenizer):
    with time_log(f"[] OpenCLIP: Embedded {len(texts)} text(s)"):
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=OPEN_CLIP_TOKEN_LENGTH_MAX,
            )
            inputs = inputs.to(DEVICE)
            text_embeddings = model.get_text_features(**inputs)
            text_embeddings = text_embeddings.cpu().numpy().tolist()
            return text_embeddings
