from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch
import numpy as np
import tqdm
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
from datasets import load_dataset, load_metric
wer_metric = load_metric("wer")

import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import json
import re

import datasets
from datasets import ClassLabel
from datasets import load_dataset, load_metric

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer

import torch
import torchaudio

import warnings
warnings.filterwarnings("ignore")
root="/kaggle/input/vlsp-asr/"
train_dataset = load_dataset("csv", data_files='/kaggle/input/vlsp-asr/VLSP/train/train_simple.csv.txt', split="train")
test_dataset = load_dataset("csv", data_files='/kaggle/input/vlsp-asr/VLSP/dev/dev_simple.csv.txt', split="train")

tokenizer = Wav2Vec2CTCTokenizer(root+"VLSP/vocab.json.txt", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# processor = Wav2Vec2Processor.from_pretrained("/home2/khanhnd/contextualized_asr/output_dir/checkpoint-500",unk_token="[unk]", pad_token="[pad]", word_delimiter_token="|")
print(processor.tokenizer.pad_token_id)
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(root+batch["audio"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["text"]
    return batch

train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names)
test_dataset = test_dataset.map(speech_file_to_array_fn, remove_columns=test_dataset.column_names)

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batch_size=8, num_proc=8, batched=True)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, batch_size=8, num_proc=8, batched=True)

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

print("LEN VOCAB", len(processor.tokenizer))
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base", 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    final_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    ctc_zero_infinity=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)-2,
    num_labels=2,
).to(device)
model.freeze_feature_extractor()

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="output_dir",
  group_by_length=True,
  per_device_train_batch_size=4,
  evaluation_strategy="steps",
  num_train_epochs=100,
  fp16=True,
  gradient_checkpointing=True, 
  save_steps=2000,
  eval_steps=2000,
  logging_steps=500,
  dataloader_num_workers=6,
  learning_rate=5e-6,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
  eval_accumulation_steps=1,
  load_best_model_at_end=True,
  greater_is_better=False,
  metric_for_best_model="wer",
  report_to='tensorboard'
)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)
print("START")
trainer.train()
# trainer.train(resume_from_checkpoint=True)