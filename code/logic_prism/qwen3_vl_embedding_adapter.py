"""Local Qwen3-VL-Embedding adapter used by the LogicAD PRISM branch.

The preprocessing, chat template, vision handling, last-token pooling, and
normalization follow the official QwenLM/Qwen3-VL-Embedding implementation.
The only project-specific extension is Matryoshka truncation: for the 8B model,
the native 4096-dimensional vector is truncated to the first 2048 dimensions
and then L2-normalized again.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from qwen_vl_utils.vision_process import process_vision_info
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import ModelOutput
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
except ImportError as exc:  # pragma: no cover - server-environment dependent
    raise ImportError(
        "Qwen3-VL-Embedding dependencies are unavailable. Use the environment "
        "from the official QwenLM/Qwen3-VL-Embedding repository, including a "
        "Transformers version with Qwen3-VL support and qwen-vl-utils."
    ) from exc

LOGGER = logging.getLogger(__name__)

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
OFFICIAL_DEFAULT_INSTRUCTION = "Represent the user's input."


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Qwen3-VL base model exposing final hidden states for last-token pooling."""

    _checkpoint_conversion_mapping: Dict[str, str] = {}
    accepts_loss_kwargs = False
    # Transformers 5.x may leave the inherited value as a class-name string.
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Qwen3VLForEmbeddingOutput:
        del logits_to_keep  # Kept only for signature compatibility.
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def _is_image_path(path: str) -> bool:
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
    clean_path = urlparse(path).path if path.startswith(("http://", "https://")) else path
    return os.path.splitext(clean_path.lower())[1] in extensions


def _is_video_input(video: Any) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and video:
        first = video[0]
        return isinstance(first, Image.Image) or (isinstance(first, str) and _is_image_path(first))
    return False


def _sample_frames(frames: List[Union[str, Image.Image]], max_segments: int):
    if len(frames) <= max_segments:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_segments, dtype=int)
    return [frames[i] for i in indices.tolist()]


class Qwen3VLEmbedder:
    """Frozen Qwen3-VL embedder with post-pooling MRL truncation."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int = 2048,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = OFFICIAL_DEFAULT_INSTRUCTION,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = None,
    ):
        model_name_or_path = os.path.abspath(model_name_or_path)
        if not os.path.isdir(model_name_or_path):
            raise FileNotFoundError(f"Qwen3-VL-Embedding checkpoint not found: {model_name_or_path}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

        self.model_name_or_path = model_name_or_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dim = int(output_dim)
        self.native_embedding_dim: Optional[int] = None
        self.max_length = int(max_length)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        self.total_pixels = int(total_pixels)
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self.default_instruction = default_instruction

        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = Qwen3VLForEmbedding.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path,
            padding_side="right",
        )

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[Any] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        effective_instruction = instruction or self.default_instruction
        effective_instruction = effective_instruction.strip()
        if effective_instruction and not unicodedata.category(effective_instruction[-1]).startswith("P"):
            effective_instruction += "."

        content: List[Dict[str, Any]] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": effective_instruction}]},
            {"role": "user", "content": content},
        ]

        texts = [] if text is None else ([text] if isinstance(text, str) else text)
        images = [] if image is None else (image if isinstance(image, list) else [image])
        if video is None:
            videos = []
        elif _is_video_input(video):
            videos = [video]
        else:
            videos = video

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for vid in videos:
            if isinstance(vid, list):
                sampled = _sample_frames(vid, max_frames or self.max_frames)
                video_content = [("file://" + x if isinstance(x, str) else x) for x in sampled]
                content.append({"type": "video", "video": video_content, "total_pixels": self.total_pixels})
            elif isinstance(vid, str):
                video_content = vid if vid.startswith(("http://", "https://")) else "file://" + vid
                content.append({
                    "type": "video",
                    "video": video_content,
                    "fps": fps or self.fps,
                    "max_frames": max_frames or self.max_frames,
                })
            else:
                raise TypeError(f"Unsupported video input type: {type(vid)}")

        for img in images:
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(("http://", "https://")) else "file://" + img
            else:
                raise TypeError(f"Unsupported image input type: {type(img)}")
            content.append({
                "type": "image",
                "image": image_content,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })

        for txt in texts:
            content.append({"type": "text", "text": txt})
        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None

        return self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def _pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_positions = attention_mask.shape[1] - attention_mask.flip(1).argmax(1) - 1
        rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[rows, last_positions]

    @torch.no_grad()
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        if not inputs:
            return torch.empty((0, self.output_dim), device=self.device, dtype=torch.float32)
        conversations = [
            self.format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                video=item.get("video"),
                instruction=item.get("instruction"),
                fps=item.get("fps"),
                max_frames=item.get("max_frames"),
            )
            for item in inputs
        ]
        processed = self._preprocess_inputs(conversations)
        processed = {key: value.to(self.device) for key, value in processed.items()}
        outputs = self.model(**processed)
        embeddings = self._pool_last(outputs.last_hidden_state, processed["attention_mask"])

        native_dim = int(embeddings.shape[-1])
        self.native_embedding_dim = native_dim
        if self.output_dim > native_dim:
            raise ValueError(
                f"Requested output_dim={self.output_dim}, but checkpoint emits {native_dim} dimensions"
            )
        # The official local wrapper has no output_dim argument. Apply MRL only
        # after pooling: retain the first dimensions, then normalize that subspace.
        embeddings = embeddings[:, : self.output_dim].float()
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
