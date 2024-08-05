import torch

from models.constants import (
    AestheticScoreResult,
    AestheticsScorer,
    OpenCLIP,
    DEVICE_CPU,
)
from .model import preprocess


def normalize(value, range_min, range_max):
    # Ensure the range is valid
    if range_min == range_max:
        raise ValueError("Minimum and maximum range values cannot be the same.")
    if range_min > range_max:
        raise ValueError(
            "Minimum range value cannot be greater than the maximum range value."
        )

    # Normalize the value
    normalized_value = (value - range_min) / (range_max - range_min)
    return max(0, min(normalized_value, 1))  # Clamp between 0 and 1


def generate_aesthetic_scores(
    image, aesthetics_scorer: AestheticsScorer, clip: OpenCLIP
) -> AestheticScoreResult:
    clip_processor = clip.processor
    vision_model = clip.model.vision_model
    rating_model = aesthetics_scorer.rating_model
    artifacts_model = aesthetics_scorer.artifacts_model

    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE_CPU)
    with torch.no_grad():
        vision_output = vision_model(**inputs)
    pooled_output = vision_output.pooler_output
    embedding = preprocess(pooled_output)
    with torch.no_grad():
        rating = rating_model(embedding)
        artifact = artifacts_model(embedding)
    return AestheticScoreResult(
        rating_score=normalize(rating.detach().cpu().item(), 0, 10),
        artifact_score=normalize(artifact.detach().cpu().item(), 0, 5),
    )
