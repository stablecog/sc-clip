import os
import traceback

from flask import Flask, request, current_app, jsonify
from waitress import serve

from models.aesthetics_scorer.main import generate_aesthetic_scores
from models.open_clip.main import (
    embeds_of_texts,
    embeds_of_images,
)
from models.constants import ModelsPack
from utils.helpers import download_images, time_log
import time
import logging
from typing import List

clipapi = Flask(__name__)


class ObjectForEmbedding:
    def __init__(self, item: any, index: int):
        self.item = item
        self.index = index


@clipapi.route("/", methods=["GET"])
def health():
    return "OK", 200


@clipapi.route("/health", methods=["GET"])
def health():
    return "OK", 200


@clipapi.route("/embed", methods=["POST"])
def clip_embed():
    s = time.time()
    with current_app.app_context():
        models_pack: ModelsPack = current_app.models_pack
    authheader = request.headers.get("Authorization")
    if authheader is None:
        logging.error("📎 🔴 Unauthorized: Missing authorization header")
        return "Unauthorized", 401
    if authheader != os.environ["CLIPAPI_AUTH_TOKEN"]:
        logging.error("📎 🔴 Unauthorized: Invalid authorization header")
        return "Unauthorized", 401

    req_body = None
    try:
        req_body = request.get_json()
    except Exception as e:
        tb = traceback.format_exc()
        logging.info(f"📎 🔴 Error parsing request body: {tb}\n")
        return str(e), 400
    finally:
        if req_body is None:
            logging.error("📎 🔴 Missing request body")
            return "Missing request body", 400
        if isinstance(req_body, list) is not True:
            logging.error("📎 🔴 Body should be an array")
            return "Body should be an array", 400

    embeds = [None for _ in range(len(req_body))]
    text_objects: List[ObjectForEmbedding] = []
    image_objects: List[ObjectForEmbedding] = []
    for index, item in enumerate(req_body):
        if "text" in item:
            text_objects.append(ObjectForEmbedding(item, index))
        if "image" in item:
            image_objects.append(ObjectForEmbedding(item, index))

    if len(text_objects) > 0:
        texts = [obj.item["text"] for obj in text_objects]
        text_embeds = embeds_of_texts(
            texts,
            models_pack.open_clip.model,
            models_pack.open_clip.tokenizer,
        )
        for i, embed in enumerate(text_embeds):
            item = text_objects[i].item
            index = text_objects[i].index
            id = item.get("id", None)
            obj = {"input_text": item["text"], "embedding": embed}
            if id is not None:
                obj["id"] = id
            embeds[index] = obj

    if len(image_objects) > 0:
        image_urls = []
        pil_images = []
        for obj in image_objects:
            image_urls.append(obj.item["image"])
        try:
            with time_log(f"📎 Downloaded {len(image_urls)} image(s)"):
                pil_images = download_images(urls=image_urls, max_workers=25)
        except Exception as e:
            tb = traceback.format_exc()
            logging.info(f"📎 🔴 Failed to download images: {tb}\n")
            return str(e), 500
        image_embeds, vision_output = embeds_of_images(
            pil_images,
            models_pack.open_clip.model,
        )
        for i, embed in enumerate(image_embeds):
            item = image_objects[i].item
            index = image_objects[i].index
            id = item.get("id", None)
            obj = {"image": image_urls[i], "embedding": embed}
            if id is not None:
                obj["id"] = id

            # Calculate score if needed
            if "calculate_score" in item and (
                item["calculate_score"] is True
                or item["calculate_score"] == "true"
                or item["calculate_score"] == "True"
            ):
                s_aes = time.time()
                pooler_output = vision_output.pooler_output[i].unsqueeze(0)
                score = generate_aesthetic_scores(
                    image=pil_images[i],
                    aesthetics_scorer=models_pack.aesthetics_scorer,
                    clip=models_pack.open_clip,
                    pooler_output=pooler_output,
                )
                obj["aesthetic_score"] = {
                    "rating": score.rating_score,
                    "artifact": score.artifact_score,
                }
                e_aes = time.time()
                logging.info(
                    f"🎨 Image {i+1} | Duration: {(e_aes - s_aes)*1000 :.0f} ms | Rating Score: {score.rating_score:.2f} | Artifact Score: {score.artifact_score:.2f}"
                )

            embeds[index] = obj

    e = time.time()
    logging.info(f"📎 ✅ Responded for {len(req_body)} item(s) in: {e-s:.2f} sec.")
    return jsonify({"embeddings": embeds})


def run_clipapi(models_pack: ModelsPack):
    host = os.environ.get("CLIPAPI_HOST", "0.0.0.0")
    port = os.environ.get("CLIPAPI_PORT", 13339)
    with clipapi.app_context():
        current_app.models_pack = models_pack
    logging.info("//////////////////////////////////////////////////////////////////")
    logging.info(f"📎 🟢 Starting CLIP API on {host}:{port}")
    logging.info("//////////////////////////////////////////////////////////////////")
    serve(clipapi, host=host, port=port)
