import uuid
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


# ------------------------
# 4. Multi-Source Distillation
# ------------------------
class TrajectoryCollector:
    def __init__(
        self,
        llm: HFModel,
        rl_trainer: RLVRTrainer,
        tree_manager: TreeManager,
        k_base: int = 256,
    ):
        self.llm = llm
        self.rl_trainer = rl_trainer
        self.tree_manager = tree_manager
        self.k_base = k_base

    def collect_base(self, prompt: str) -> List[Dict[str, Union[str, List[str]]]]:
        recs = []
        for _ in range(self.k_base):
            out = self.llm.generate(prompt, do_sample=True, temperature=0.8)
            recs.append({"input": prompt, "cot": None, "output": out, "source": "base"})
        return recs

    def collect_rl(self, prompt: str) -> Dict[str, Union[str, List[str]]]:
        self.rl_trainer.train_step([prompt])
        _, seqs, _, _ = self.llm.generate_with_values([prompt])
        out = self.llm.tokenizer.decode(seqs[0], skip_special_tokens=True)
        return {"input": prompt, "cot": None, "output": out, "source": "rlvr"}

    def collect_tot(
        self, prompt: str, strategy: str = "best_first"
    ) -> Dict[str, Union[str, List[str]]]:
        self.tree_manager.initialize(prompt)
        final = self.tree_manager.traverse(strategy)
        # reconstruct path
        path = []
        node = final
        while node:
            path.append(node.state)
            node = node.parent
        return {
            "input": prompt,
            "cot": list(reversed(path)),
            "output": final.state,
            "source": "tot",
        }


class StudentDistiller:
    def __init__(
        self,
        student_model_name: str,
        trajectories: List[Dict[str, Any]],
        output_dir: str = "./distilled",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(student_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(student_model_name)
        examples = []
        for rec in trajectories:
            cot = ""
            if rec["cot"]:
                cot = "\n".join(rec["cot"]) + "\n"
            inp, out = rec["input"], rec["output"]
            examples.append({"input_text": inp, "target_text": cot + out})

        # prepare dataset dict
        def preprocess(x):
            enc = self.tokenizer(
                x["input_text"], truncation=True, padding="max_length", max_length=128
            )
            dec = self.tokenizer(
                x["target_text"], truncation=True, padding="max_length", max_length=128
            )
            enc["labels"] = dec["input_ids"]
            return enc

        self.dataset = [preprocess(ex) for ex in examples]
        self.args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            save_steps=500,
            logging_steps=100,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
        )

    def fine_tune(self):
        self.trainer.train()
        self.trainer.save_model()
