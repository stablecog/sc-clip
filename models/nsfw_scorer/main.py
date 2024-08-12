from typing import List
from PIL import Image

from models.constants import NSFWScoreResult, NSFWScorer


def generate_nsfw_score(
    images: List[Image.Image], nsfw_scorer: NSFWScorer
) -> List[NSFWScoreResult]:
    pipeline = nsfw_scorer.pipeline
    results = pipeline(images)
    scores: List[NSFWScoreResult] = []
    for result in results:
        nsfw_score = next(
            (item["score"] for item in result if item["label"] == "nsfw"), None
        )
        if nsfw_score is None:
            raise ValueError("NSFW label not found in the result.")
        scores.append(NSFWScoreResult(nsfw_score=nsfw_score))
    return scores
