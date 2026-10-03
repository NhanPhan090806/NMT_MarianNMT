"""Offline translation from a short, self-contained export directory."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .config import DataConfig, ModelConfig
from .data import ScratchTokenizer, load_pretrained_tokenizer, normalize
from .models import MarianTranslator, build_model, generate


class Translator:
    def __init__(self, model, tokenizer, data_config, device=None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model, self.tokenizer, self.config = model.to(self.device).eval(), tokenizer, data_config

    @classmethod
    def from_export(cls, directory, device=None):
        directory = Path(directory).expanduser().resolve()
        settings = json.loads((directory / "translation_config.json").read_text(encoding="utf-8"))
        model_config, data_config = ModelConfig(**settings["model"]), DataConfig(**settings["data"])
        if model_config.stage == "marian_en_vi":
            from transformers import MarianMTModel
            tokenizer = load_pretrained_tokenizer(directory, local_files_only=True)
            model = MarianTranslator(model_config, MarianMTModel.from_pretrained(directory, local_files_only=True))
        else:
            from safetensors.torch import load_file
            tokenizer = ScratchTokenizer(directory)
            model = build_model(model_config, SimpleNamespace(tokenizer=tokenizer, config=data_config, pretrained=False))
            model.load_state_dict(load_file(str(directory / "model.safetensors")))
        return cls(model, tokenizer, data_config, device)

    @torch.inference_mode()
    def translate(self, text):
        text = normalize(text)
        if not text:
            raise ValueError("Enter a nonempty English sentence.")
        if isinstance(self.model, MarianTranslator):
            ids = self.tokenizer(self.config.target_prefix + text, truncation=False, verbose=False)["input_ids"]
        else:
            ids = self.tokenizer.encode(text)
        if len(ids) > self.config.max_input_length:
            raise ValueError("Source exceeds the saved input cap; shorten it. No text was truncated.")
        batch = {"input_ids": torch.tensor([ids], device=self.device),
                 "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=self.device)}
        output = generate(self.model, batch, self.config.max_output_length)
        return self.tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)[0].strip()
