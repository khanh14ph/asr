from multiprocessing import get_context
import torchaudio
from datasets import load_dataset
from jiwer import wer
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
)
from model import Wav2Vec2ForCTCKan
import kenlm
from pyctcdecode import build_ctcdecoder
import torch
import numpy as np
import pandas as pd
import random
import os

device = "cuda"
from glob import glob

# lst=["type_4_averaged_last_5",
#     "type_4_averaged_last_10",
# "type_4_averaged_last_30"]
# load pretrained model
# for file_path in lst:
#     typ,name=file_path.split("_averaged_")
vocab_file = "/home4/khanhnd/thesis/asr_base/vocab.json"
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file,
    # "/home4/khanhnd/thesis/asr_base/vocab.json" ,
    bos_token=None,
    eos_token=None,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
    replace_word_delimiter_char=" ",
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
import glob

import json

# Opening JSON file
with open(vocab_file) as json_file:

    my_dict = json.load(json_file)

my_dict = processor.tokenizer.get_vocab()
labels = sorted(my_dict, key=lambda k: my_dict[k])
# print(labels)
# prepare decoder and decode logits via shallow fusion
alpha = 1
beta = 1
kenlm_model_path="/home4/khanhnd/improved_unimatch/language_model/base_selection.arpa"
# kenlm_model_path="/home4/khanhnd/model/vi_lm_5grams.bin"
# kenlm_model_path="/home4/khanhnd/improved_unimatch/language_model/aicc_train.arpa"
decoder = build_ctcdecoder(
    labels,
   kenlm_model_path=kenlm_model_path,
    alpha=alpha,
    beta=beta,
)
test_dataset = load_dataset(
    "csv",
    data_files="/home4/khanhnd/asr-phoneme/predict_test_char.csv",
    # data_files="/home4/khanhnd/train_b2.csv",
    # data_files="/home4/khanhnd/improved_unimatch/csv/vlsp_2020_test_task1.csv",
    sep=",",
    split="train",
    cache_dir="/home3/khanhnd/cache",
)

# test_dataset = test_dataset.select([i for i in range(1000)])

# random_indices = [i for i in range(1000)]
# test_dataset = test_dataset.select(random_indices)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio_path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["input_length"] = len(batch["speech"]) / sampling_rate
    if batch["transcription"] == None:

        batch["transcription"] = ""
    else:
        batch["transcription"] = batch["transcription"]
    batch.pop("phoneme_transcript")
    batch.pop("accent")
    return batch

# test_dataset = test_dataset.select([i for i in range(60000,80000)])
test_dataset1 = test_dataset.map(speech_file_to_array_fn, num_proc=8)

from torch import nn
import time

# while os.path.exists("/home4/khanhnd/improved_unimatch/checkpoint/fixmatch_onl/checkpoint-6000")==False:
#     time.sleep(1)
#     print("waiting...")
final_checkpoints = (
    # "/home4/khanhnd/improved_unimatch/checkpoint/base/checkpoint-125010",
    # "/home4/khanhnd/improved_unimatch/checkpoint/aicc/checkpoint-168000",
                #    "/home3/khanhnd/base+tts/checkpoint-128000",
                #    "/home3/khanhnd/base_whisper/checkpoint-140000",
                #    "/home3/khanhnd/multitask/checkpoint-86000/",
                # "/home4/khanhnd/improved_unimatch/checkpoint/fixmatch_off/checkpoint-150000",
                   "/home3/khanhnd/fixmatch_tts_per_no_adver_2/checkpoint-103500/",
                #    "/home3/khanhnd/fixmatch_tts_per_no_adver_3/checkpoint-54000",
                    # "/home4/khanhnd/improved_unimatch/checkpoint/fixmatch_vanila_no_tts/checkpoint-20500",
                #     "/home4/khanhnd/improved_unimatch/checkpoint/base/checkpoint-125010",
                # "/home4/khanhnd/improved_unimatch/checkpoint/topline_continue/checkpoint-159200",
                # "/home4/khanhnd/improved_unimatch/checkpoint/fixmatch_vanila_no_tts/checkpoint-92000",
                # "/home4/khanhnd/fixmatch_vanila/checkpoint-70500"
                # "/home3/khanhnd/vin_bigdata_sample/checkpoint-104000",
                # "/home3/khanhnd/youtube_sample/checkpoint-104000",
                # "/home3/khanhnd/youtube_sample/checkpoint-110000",
                # "/home3/khanhnd/vietbud_sample/checkpoint-202000",
                # "/home3/khanhnd/base+tts/checkpoint-128000",
                # "/home3/khanhnd/youtube_sample_1/checkpoint-20000",



)
import pandas as pd

# from normalize import normalizetmu

from torchmetrics.text import WordErrorRate

wer = WordErrorRate()


def filter(pred_lst, ref):
    min_wer = 1000
    res = ""
    for i in pred_lst:
        temp_wer = wer(i, ref)
        if temp_wer < min_wer:
            min_wer = temp_wer
            res = i
    return res


for final_checkpoint in final_checkpoints:
    print("Load from ", final_checkpoint)

    model = Wav2Vec2ForCTC.from_pretrained(final_checkpoint).eval().to(device)

    def map_to_pred(batch, pool):
        inputs = processor(
            batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        logits_list = logits.cpu().numpy()


        res = decoder.decode_beams_batch(
            logits_list=logits_list, pool=pool, beam_width=100
        )
        text_lst = [[j[0] for j in i] for i in res]
        # best_text = [filter(i, j) for i, j in zip(text_lst, batch["transcription"])]
        best_text = [i[0][0] for i in res]
        batch["pred_text"] = best_text
        # print(best_text)
        # print(batch["transcription"])
        # print("_____________")
        # batch["transcription"] = text
        # predicted_ids = torch.argmax(logits, dim=-1)
        # batch["pred_text"] = processor.batch_decode(predicted_ids)

        batch.pop("speech")

        batch.pop("sampling_rate")
        batch.pop("input_length")
        return batch

    s = time.time()
    with get_context("fork").Pool(processes=4) as pool:
        result = test_dataset1.map(
            map_to_pred, batched=True, batch_size=4, fn_kwargs={"pool": pool}
        )

    result.set_format(
        type="pandas",
        columns=["audio_path", "pred_text", "transcription"],
    )
    # name = final_checkpoint.split("/")[-2]
    name = "temp.csv"
    save_path = f"/home4/khanhnd/improved_unimatch/result/proposed.csv"
    result.to_csv(save_path, index=False, sep="|")
    with open("/home4/khanhnd/improved_unimatch/data_synthetic/pseudo_test.txt", "w") as f:
        for i in list(result["pred_text"]):
            f.write(i+"\n")
    df = pd.read_csv(save_path, sep="|")
    # df = df[["audio_path", "pred_text", "transcription", "score"]]

    df = df.fillna("")
    e = time.time()

    with open("/home4/khanhnd/improved_unimatch/result.txt", "a") as f:
        f.write(final_checkpoint+"|" +kenlm_model_path+"|"+ str(wer(list(df["pred_text"]), list(df["transcription"])))+"\n")
