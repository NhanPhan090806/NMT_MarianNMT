"""Offline behavioral tests for the new translation pipeline."""

import json
from dataclasses import replace

import pytest
import torch
from transformers import MarianConfig, MarianMTModel

from nmt_assignment.config import DataConfig, ModelConfig, STAGES, TrainConfig
from nmt_assignment.data import TranslationCollator, normalize, prepare_data, split_corpus
from nmt_assignment.evaluation import evaluate, translation_metrics
from nmt_assignment.inference import Translator
from nmt_assignment.models import MarianTranslator, build_model, generate
from nmt_assignment.training import TranslationTrainer
from nmt_assignment.workflow import (archive_translation_outputs, collect_results,
                                     learning_diagnostic, run_experiment)
from qa_assignment.training import load_checkpoint
from qa_assignment.utils import seed_everything


class FakePretrainedTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def __len__(self):
        return 64

    def __call__(self, texts=None, text_target=None, **kwargs):
        values = text_target if text_target is not None else texts
        if isinstance(values, str):
            return {"input_ids": [2 + len(w) % 60 for w in values.split()] + [1]}
        if text_target is None:
            assert all(v.startswith(">>vie<< ") for v in values)
        return {"input_ids": [[2 + len(w) % 60 for w in v.split()] + [1] for v in values]}

    def batch_decode(self, values, skip_special_tokens=True):
        return [" ".join(f"w{int(t)}" for t in row if int(t) not in (0, 1)) for row in values]

    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer_config.json").write_text('{"test":true}', encoding="utf-8")


@pytest.fixture
def nmt_bundle(tmp_path):
    train = [{"translation": {"en": f"I see {i} birds in the sky .", "vi": f"Tôi thấy {i} con chim trên trời ."}} for i in range(14)]
    test = [{"translation": {"en": "The weather is nice today .", "vi": "Hôm nay thời tiết đẹp ."}},
            {"translation": {"en": "Scientists study the changing climate .", "vi": "Các nhà khoa học nghiên cứu khí hậu ."}}]
    config = DataConfig(validation_examples=3, max_train_examples=7, vocab_size=128,
                        max_input_length=48, max_output_length=48, max_words=30)
    return prepare_data(tmp_path / "data", config, raw_splits={"train": train, "test": test},
                        pretrained_tokenizer=FakePretrainedTokenizer())


def small_model(stage):
    return ModelConfig(stage=stage, d_model=16, encoder_layers=1, decoder_layers=1,
                       heads=2, feedforward_dim=32, dropout=0.1)


def tiny_training():
    return TrainConfig(epochs=2, micro_batch_size=2, accumulation_steps=3, eval_batch_size=2,
                       save_every_steps=1, log_every_steps=1, keep_step_checkpoints=10,
                       development_limit=None, early_stopping_patience=0)


def test_normalization_splits_and_source_leakage():
    assert normalize("  Tôi &amp; bạn\n") == "Tôi & bạn"
    pair = lambda en, vi="một câu": {"translation": {"en": en, "vi": vi}}
    train = [pair(f"source {i}") for i in range(12)] + [pair("source 1", "bản khác"), pair("test source")]
    splits, audit = split_corpus({"train": train, "test": [pair("test source")]},
                                DataConfig(validation_examples=3, max_train_examples=5))
    sources = [{r["source"] for r in splits[k]} for k in ("train", "validation", "test")]
    assert not sources[0] & sources[1] and not sources[0] & sources[2] and not sources[1] & sources[2]
    assert audit["train_filtering"]["duplicate_source"] == 1
    assert audit["train_test_source_overlap_removed"] == 1
    assert [len(s) for s in sources] == [5, 3, 1]


def test_cache_tokenizer_and_identical_stage_scope(nmt_bundle):
    config = nmt_bundle.config
    cached = prepare_data(nmt_bundle.directory.parents[1], config,
                          pretrained_tokenizer=FakePretrainedTokenizer())
    assert cached.manifest == nmt_bundle.manifest
    pretrained = nmt_bundle.for_pretrained()
    for scratch, other in ((nmt_bundle.train, pretrained.train), (nmt_bundle.development, pretrained.development),
                            (nmt_bundle.test, pretrained.test)):
        assert [r["id"] for r in scratch.rows] == [r["id"] for r in other.rows]
        assert [r["target"] for r in scratch.rows] == [r["target"] for r in other.rows]
    assert pretrained.manifest["data_fingerprint"] == nmt_bundle.manifest["data_fingerprint"]
    # Train-only SentencePiece sees these selected rows, not held-out text.
    ids = {r["id"] for r in nmt_bundle.train.rows}
    assert not ids & {r["id"] for r in nmt_bundle.development.rows + nmt_bundle.test.rows}
    manifest = nmt_bundle.directory / "manifest.json"
    (nmt_bundle.directory / "test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        prepare_data(nmt_bundle.directory.parents[1], config, pretrained_tokenizer=FakePretrainedTokenizer())
    assert manifest.is_file()


def test_dual_tokenizer_limits_never_silently_truncate(tmp_path):
    from nmt_assignment.data import encode_splits, train_tokenizer
    rows = [{"id": "1", "source": "a short sentence", "target": "một câu ngắn"}]
    scratch = train_tokenizer(rows, tmp_path / "tokenizer", 64)
    class TooLong(FakePretrainedTokenizer):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result["input_ids"] = [ids + [2] * 100 for ids in result["input_ids"]]
            return result
    with pytest.raises(ValueError, match="fit both tokenizers"):
        encode_splits({"train": rows}, scratch, TooLong(), DataConfig(max_input_length=32, max_output_length=32))


@pytest.mark.parametrize("stage", STAGES[:-1])
def test_scratch_causality_padding_and_cached_generation(nmt_bundle, stage):
    seed_everything(9)
    model = build_model(small_model(stage), nmt_bundle).eval()
    batch = TranslationCollator(nmt_bundle.tokenizer)(nmt_bundle.train.rows[:2])
    with torch.no_grad():
        first = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
        changed = batch["labels"].clone()
        changed[:, 2:] = 5
        later = model(batch["input_ids"], batch["attention_mask"], changed)
        torch.testing.assert_close(first[:, :3], later[:, :3], rtol=1e-5, atol=1e-6)
        padded_inputs = torch.nn.functional.pad(batch["input_ids"], (0, 2), value=0)
        padded_mask = torch.nn.functional.pad(batch["attention_mask"], (0, 2), value=0)
        padded = model(padded_inputs, padded_mask, batch["labels"])
        torch.testing.assert_close(first, padded, rtol=1e-4, atol=1e-6)
        tokens = torch.tensor([[3, 7, 8], [3, 9, 10]])
        memory = model.encode(batch["input_ids"], batch["attention_mask"])
        full, _ = model.decode_tokens(tokens, memory)
        cache, pieces = None, []
        for token in tokens.split(1, dim=1):
            logits, cache = model.decode_tokens(token, memory, cache=cache, use_cache=True)
            pieces.append(logits)
        torch.testing.assert_close(full, torch.cat(pieces, 1), rtol=1e-4, atol=1e-6)
    assert first.shape[:2] == batch["labels"].shape


@pytest.mark.parametrize("stage", STAGES[:-1])
def test_optimizer_boundary_resume_matches_uninterrupted(nmt_bundle, tmp_path, stage):
    seed_everything(17)
    model = build_model(small_model(stage), nmt_bundle)
    collator = TranslationCollator(nmt_bundle.tokenizer)
    config = tiny_training()
    reference = TranslationTrainer(model, nmt_bundle, collator, config, tmp_path / "reference", "cpu")
    reference.fit()
    checkpoint = tmp_path / "reference" / "step_00000001.pt"
    assert load_checkpoint(checkpoint)["state"]["next_batch"] == 3
    resumed_model = build_model(small_model(stage), nmt_bundle)
    resumed = TranslationTrainer(resumed_model, nmt_bundle, collator, config, tmp_path / "resumed", "cpu")
    resumed.resume(checkpoint)
    resumed.fit()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
    assert reference.state["global_step"] == resumed.state["global_step"] == 4
    bad = TranslationTrainer(resumed_model, nmt_bundle, collator, replace(config, learning_rate=2e-3),
                             tmp_path / "bad", "cpu")
    with pytest.raises(ValueError, match="differ"):
        bad.resume(checkpoint)


def test_translation_signatures_and_plateau_stopping(nmt_bundle, tmp_path):
    sentence = "Tôi thích nghiên cứu những hệ thống phức tạp ."
    scores = translation_metrics([sentence], [sentence])
    assert scores["bleu"] == pytest.approx(100) and scores["chrf"] == pytest.approx(100)
    assert "tok:13a" in scores["bleu_signature"]
    model = build_model(small_model("rnn"), nmt_bundle)
    config = replace(tiny_training(), early_stopping_patience=3, early_stopping_min_delta=0.2,
                     early_stopping_min_epochs=1)
    trainer = TranslationTrainer(model, nmt_bundle, TranslationCollator(nmt_bundle.tokenizer), config,
                                 tmp_path / "stop", "cpu")
    for step, score in enumerate((30.0, 29.9, 30.1, 30.0), start=1):
        trainer.state["global_step"] = step
        trainer.update_early_stopping(score)
    assert trainer.state["early_stopping"]["stopped"]
    assert trainer.state["early_stopping"]["stop_step"] == 4


def test_checkpoint_selection_uses_actual_best_chrf_not_last_or_training_loss(nmt_bundle, tmp_path, monkeypatch):
    from nmt_assignment import training
    checks = iter([(20.0, 5.0), (19.0, 4.0), (20.1, 4.5), (20.0, 4.0)])
    def fake_evaluation(*args, **kwargs):
        score, loss = next(checks)
        return {"chrf": score, "bleu": 1.0, "teacher_forced_loss": loss, "token_accuracy": 10.0, "samples": []}
    monkeypatch.setattr(training, "evaluate", fake_evaluation)
    config = replace(tiny_training(), early_stopping_patience=3, early_stopping_min_delta=0.2,
                     early_stopping_min_epochs=1)
    trainer = TranslationTrainer(build_model(small_model("rnn"), nmt_bundle), nmt_bundle,
                                 TranslationCollator(nmt_bundle.tokenizer), config, tmp_path / "select", "cpu")
    for step in range(1, 5):
        trainer.state["global_step"] = step
        trainer.development_evaluation()
    selected = load_checkpoint(tmp_path / "select" / "best.pt")
    assert selected["state"]["global_step"] == 3 and selected["state"]["best_chrf"] == 20.1
    assert trainer.state["early_stopping"]["stopped"]


@pytest.mark.parametrize("stage", STAGES[:-1])
def test_end_to_end_export_and_offline_translation(nmt_bundle, tmp_path, stage):
    output = tmp_path / "nmt"
    config = replace(tiny_training(), epochs=1, accumulation_steps=2)
    result = run_experiment(nmt_bundle, small_model(stage), config, output, device="cpu", benchmark_examples=2)
    assert result["test"]["examples"] == len(nmt_bundle.test)
    assert "bleu" in result["test"] and "chrf" in result["test"]
    assert result["benchmark"]["throughput"]["sentences_per_second"] > 0
    assert (output / "checkpoints" / stage / "seed_42" / "last.pt").is_file()
    assert (output / "checkpoints" / stage / "seed_42" / "best.pt").is_file()
    directory = output / "models" / stage / "seed_42"
    translator = Translator.from_export(directory, device="cpu")
    assert isinstance(translator.translate("I see birds ."), str)
    with pytest.raises(ValueError, match="No text was truncated"):
        translator.translate("birds " * 100)
    assert collect_results(output).stage.tolist() == [stage]
    assert archive_translation_outputs(output).is_file()
    resumed = run_experiment(nmt_bundle, small_model(stage), config, output, device="cpu", benchmark_examples=0)
    assert resumed["optimizer_steps"] == result["optimizer_steps"]


def test_diagnostic_has_only_training_sources(nmt_bundle, tmp_path):
    report = learning_diagnostic(nmt_bundle, small_model("transformer"), tmp_path,
                                steps=3, examples=3, check_every=1)
    assert not report["overfit_demonstrated"]
    assert {p["id"] for p in report["final"]["predictions"]} <= {r["id"] for r in nmt_bundle.train.rows}
    assert len(report["history"]) == 3


def test_pretrained_marian_forward_and_generation(nmt_bundle):
    bundle = nmt_bundle.for_pretrained()
    hf = MarianMTModel(MarianConfig(vocab_size=64, decoder_vocab_size=64, d_model=16,
        encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, pad_token_id=0, eos_token_id=1,
        decoder_start_token_id=0, max_position_embeddings=64, forced_eos_token_id=1))
    model = MarianTranslator(small_model("marian_en_vi"), hf)
    collator = TranslationCollator(bundle.tokenizer)
    metrics = evaluate(model, bundle.test, collator, bundle.tokenizer, "cpu", 8, batch_size=2)
    assert metrics["examples"] == 2 and metrics["teacher_forced_loss"] > 0
    with pytest.raises(ValueError, match="own pretrained tokenizer"):
        build_model(small_model("marian_en_vi"), nmt_bundle)


def test_local_marian_tokenizer_unicode_paths_and_portable_reload(nmt_bundle, tmp_path):
    from nmt_assignment.data import load_pretrained_tokenizer
    folder = tmp_path / "tiếng_việt_tokenizer"
    folder.mkdir()
    proto = (nmt_bundle.tokenizer.directory / "joint.model").read_bytes()
    (folder / "source.spm").write_bytes(proto)
    (folder / "target.spm").write_bytes(proto)
    vocab = {nmt_bundle.tokenizer.processor.id_to_piece(i): i for i in range(len(nmt_bundle.tokenizer))}
    vocab[">>vie<<"] = len(vocab)
    (folder / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (folder / "tokenizer_config.json").write_text('{"tokenizer_class":"MarianTokenizer"}', encoding="utf-8")
    tokenizer = load_pretrained_tokenizer(folder, local_files_only=True)
    assert ">>vie<<" in tokenizer.supported_language_codes
    ids = tokenizer(">>vie<< I see birds .")["input_ids"]
    exported = tmp_path / "xuất_mô_hình"
    tokenizer.save_pretrained(exported)
    reloaded = load_pretrained_tokenizer(exported, local_files_only=True)
    assert reloaded(">>vie<< I see birds .")["input_ids"] == ids


def test_pretrained_training_checkpoint_export_and_offline_reload(nmt_bundle, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from nmt_assignment import workflow
    bundle = nmt_bundle.for_pretrained()
    raw = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2,
                              **{f"w{i}": i for i in range(3, 64)}}, unk_token="<unk>"))
    bundle.tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>",
                                              eos_token="</s>", unk_token="<unk>")
    def tiny_pretrained(config, prepared):
        hf = MarianMTModel(MarianConfig(vocab_size=64, decoder_vocab_size=64, d_model=16,
            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=32, decoder_ffn_dim=32, pad_token_id=0, eos_token_id=1,
            decoder_start_token_id=0, max_position_embeddings=64, forced_eos_token_id=1))
        return MarianTranslator(config, hf)
    monkeypatch.setattr(workflow, "build_model", tiny_pretrained)
    output = tmp_path / "pretrained"
    summary = run_experiment(bundle, small_model("marian_en_vi"), replace(tiny_training(), epochs=1),
                             output, device="cpu", benchmark_examples=0)
    assert summary["pretrained"] and summary["test"]["examples"] == 2
    translator = Translator.from_export(output / "models" / "marian_en_vi" / "seed_42", device="cpu")
    assert isinstance(translator.translate("a sentence"), str)
    assert (output / "checkpoints" / "marian_en_vi" / "seed_42" / "best.pt").is_file()
