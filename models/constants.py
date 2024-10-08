from dotenv import load_dotenv
import os

load_dotenv()

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
SC_CLIP_VERSION = "v1.1"

_DEVICE = os.getenv("DEVICE", DEVICE_CPU)
DEVICE = _DEVICE if _DEVICE in [DEVICE_CPU, DEVICE_CUDA] else DEVICE_CPU


class OpenCLIP:
    def __init__(self, model, processor, tokenizer):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer


class AestheticsScorer:
    def __init__(self, rating_model, artifacts_model):
        self.rating_model = rating_model
        self.artifacts_model = artifacts_model


class NSFWScorer:
    def __init__(self, pipeline):
        self.pipeline = pipeline


class ModelsPack:
    def __init__(
        self,
        open_clip: OpenCLIP,
        aesthetics_scorer: AestheticsScorer,
        nsfw_scorer: NSFWScorer,
    ):
        self.open_clip = open_clip
        self.aesthetics_scorer = aesthetics_scorer
        self.nsfw_scorer = nsfw_scorer


class AestheticScoreResult:
    def __init__(
        self,
        rating_score: float,
        artifact_score: float,
    ):
        self.rating_score = rating_score
        self.artifact_score = artifact_score


class NSFWScoreResult:
    def __init__(
        self,
        nsfw_score: float,
    ):
        self.nsfw_score = nsfw_score
