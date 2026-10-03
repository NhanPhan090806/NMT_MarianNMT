"""Load an exported, fine-tuned T5 once and answer questions about a passage."""

import json
from pathlib import Path

import torch

from .data import input_text


def find_t5_exports(roots):
    """Discover local HF exports without downloading or loading model weights."""
    found = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for config_file in root.rglob("config.json"):
            directory = config_file.parent
            if not (directory / "tokenizer_config.json").is_file():
                continue
            if not any((directory / name).is_file() for name in (
                    "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                    "pytorch_model.bin.index.json")):
                continue
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if config.get("model_type") == "t5":
                found.add(directory.resolve())
    return sorted(found)


class T5Answerer:
    def __init__(self, model, tokenizer, max_input_length=512, max_output_length=64, device=None):
        if min(max_input_length, max_output_length) < 1:
            raise ValueError("Input and output limits must be positive.")
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_input_length, self.max_output_length = max_input_length, max_output_length
        self.history = []

    @classmethod
    def from_export(cls, directory, device=None):
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        directory = Path(directory).expanduser().resolve()
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Set MODEL_DIR to the exported hf_export/ folder: {directory}")
        model_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if model_config.get("model_type") != "t5":
            raise ValueError("The selected export is not a T5 model.")
        data_config = {}
        manifest = directory.parent / "data_manifest.json"
        if manifest.is_file():
            data_config = json.loads(manifest.read_text(encoding="utf-8"))["contract"]["config"]
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model = T5ForConditionalGeneration.from_pretrained(directory, local_files_only=True)
        if len(tokenizer) > model.config.vocab_size:
            raise ValueError("Exported tokenizer vocabulary exceeds the model's vocabulary.")
        return cls(model, tokenizer, data_config.get("max_input_length", 512),
                   data_config.get("max_output_length", 64), device)

    @torch.inference_mode()
    def answer(self, question, context, max_new_tokens=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a nonempty question.")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Provide the passage containing the answer.")
        limit = self.max_output_length if max_new_tokens is None else max_new_tokens
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= self.max_output_length:
            raise ValueError(f"max_new_tokens must be between 1 and {self.max_output_length}.")
        prompt = input_text({"question": question.strip(), "context": context.strip()})
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False, verbose=False)
        input_tokens = encoded["input_ids"].shape[1]
        if input_tokens > self.max_input_length:
            raise ValueError(f"Question plus passage has {input_tokens} tokens; limit is "
                             f"{self.max_input_length}. Shorten the passage; no text was truncated.")
        encoded = {key: value.to(self.device) for key, value in encoded.items()
                   if key in ("input_ids", "attention_mask")}
        generated = self.model.generate(**encoded, max_new_tokens=limit, do_sample=False,
                                        num_beams=1, use_cache=True)
        answer = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        self.history.append({"question": question.strip(), "context": context.strip(),
                             "answer": answer, "input_tokens": input_tokens})
        return answer

    def answer_many(self, questions, context):
        return [{"question": question, "answer": self.answer(question, context)} for question in questions]

    def interactive(self, context):
        """Optional manual notebook input loop; an empty question exits."""
        print("Ask questions about the supplied passage. Leave the question empty to finish.")
        while True:
            question = input("Question: ").strip()
            if not question:
                return
            print("Answer:", self.answer(question, context) or "<empty answer>")
