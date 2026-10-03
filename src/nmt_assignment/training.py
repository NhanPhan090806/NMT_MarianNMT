"""Translation checkpoint selection on development chrF; shared resumable optimizer loop."""

import math

from qa_assignment.training import Trainer as BaseTrainer
from qa_assignment.utils import reset_memory, write_json
from .evaluation import evaluate


class TranslationTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state.update(best_chrf=-1.0, best_development_loss=None)
        self.state["early_stopping"] = {"best_chrf": -1.0, "checks_without_improvement": 0,
                                         "stopped": False, "stop_step": None}
        self.contract["task"] = "english_to_vietnamese_translation"
        self.contract["selection_metric"] = "development_chrf"
        self.contract["tokenization"] = "marian" if self.bundle.pretrained else "train_only_sentencepiece"

    def development_evaluation(self):
        self.capture_training_memory()
        result = evaluate(self.model, self.bundle.development, self.collator, self.bundle.tokenizer,
                          self.device, self.bundle.config.max_output_length, self.config.eval_batch_size,
                          self.config.precision, self.config.development_limit)
        result.update(step=self.state["global_step"], epoch_cursor=self.state["epoch"],
                      training_seconds=self.state["training_seconds"])
        self.state["history"].append(result)
        score, loss = result["chrf"], result["teacher_forced_loss"]
        better = score > self.state["best_chrf"] or (score == self.state["best_chrf"] and (
            self.state["best_development_loss"] is None or loss < self.state["best_development_loss"]))
        if better:
            self.state["best_chrf"], self.state["best_development_loss"] = score, loss
        self.update_early_stopping(score)
        result["early_stopping"] = self.state["early_stopping"].copy()
        if better:
            self.save("best.pt")
        self.save()
        write_json(self.checkpoint_dir / "history.json", self.state["history"])
        print(f"Development step={result['step']}: BLEU={result['bleu']:.2f} chrF={score:.2f} "
              f"loss={loss:.4f} token_accuracy={result['token_accuracy']:.2f}%")
        for sample in result["samples"][:2]:
            print(f"  EN: {sample['source']}\n  VI: {sample['reference']}\n  Model: {sample['prediction']}")
        self.model.train()
        reset_memory(self.device)

    def update_early_stopping(self, score):
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite development chrF.")
        config, stopping = self.config, self.state["early_stopping"]
        if config.early_stopping_patience == 0:
            return
        if score > stopping["best_chrf"] + config.early_stopping_min_delta:
            stopping["best_chrf"], stopping["checks_without_improvement"] = score, 0
        elif (self.state["global_step"] >= config.early_stopping_min_steps and
              self.state["epoch"] + 1 >= config.early_stopping_min_epochs):
            stopping["checks_without_improvement"] += 1
        if stopping["checks_without_improvement"] >= config.early_stopping_patience:
            stopping["stopped"], stopping["stop_step"] = True, self.state["global_step"]
            print("Early stopping: development chrF plateaued. The highest-chrF checkpoint is retained.")
