import os
import time
from huggingface_hub import login
from transformers import AutoModel, AutoProcessor, AutoTokenizer, pipeline
from models.aesthetics_scorer.constants import (
    AESTHETICS_SCORER_CACHE_DIR,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_CONFIG,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_WEIGHT_URL,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_CONFIG,
    AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_WEIGHT_URL,
)
from models.aesthetics_scorer.model import load_model as load_aesthetics_scorer_model
from models.constants import (
    DEVICE,
    SC_CLIP_VERSION,
    AestheticsScorer,
    ModelsPack,
    NSFWScorer,
    OpenCLIP,
)
from models.nsfw_scorer.constants import NSFW_SCORER_MODEL_ID
from models.open_clip.constants import OPEN_CLIP_MODEL_CACHE, OPEN_CLIP_MODEL_ID
import logging
from tabulate import tabulate

from utils.logger import TabulateLevels


def setup() -> ModelsPack:
    start = time.time()
    version_str = f"Version: {SC_CLIP_VERSION}"
    logging.info(
        tabulate(
            [["🟡 Setup started", version_str]], tablefmt=TabulateLevels.PRIMARY.value
        )
    )

    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is not None:
        login(token=hf_token)
        logging.info(f"✅ Logged in to HuggingFace")

    # For OpenCLIP
    logging.info("🟡 Loading OpenCLIP")
    open_clip = OpenCLIP(
        model=AutoModel.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=OPEN_CLIP_MODEL_CACHE
        ).to(DEVICE),
        processor=AutoProcessor.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=OPEN_CLIP_MODEL_CACHE
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            OPEN_CLIP_MODEL_ID, cache_dir=OPEN_CLIP_MODEL_CACHE
        ),
    )
    logging.info("✅ Loaded OpenCLIP")

    # For asthetics scorer
    logging.info("🟡 Loading Aesthetics Scorer")
    aesthetics_scorer = AestheticsScorer(
        rating_model=load_aesthetics_scorer_model(
            weight_url=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_WEIGHT_URL,
            cache_dir=AESTHETICS_SCORER_CACHE_DIR,
            config=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_RATING_CONFIG,
        ).to(DEVICE),
        artifacts_model=load_aesthetics_scorer_model(
            weight_url=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_WEIGHT_URL,
            cache_dir=AESTHETICS_SCORER_CACHE_DIR,
            config=AESTHETICS_SCORER_OPENCLIP_VIT_H_14_ARTIFACT_CONFIG,
        ).to(DEVICE),
    )
    logging.info("✅ Loaded Aesthetics Scorer")

    # For NSFW scorer
    logging.info("🟡 Loading NSFW Scorer")
    nsfw_scorer = NSFWScorer(
        pipeline=pipeline(
            "image-classification", model=NSFW_SCORER_MODEL_ID, device=DEVICE
        )
    )
    logging.info("✅ Loaded Aesthetics Scorer")

    end = time.time()
    logging.info("//////////////////////////////////////////////////////////////////")
    logging.info(f"✅ Setup is done in: {round((end - start))} sec.")
    logging.info("//////////////////////////////////////////////////////////////////")

    return ModelsPack(
        open_clip=open_clip,
        aesthetics_scorer=aesthetics_scorer,
        nsfw_scorer=nsfw_scorer,
    )
