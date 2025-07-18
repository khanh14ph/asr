import os 
import collections
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config

# checkpoints_paths = ['/home4/tuannd/vbee-asr/asr-experiment/models/huggingface/address_vc_weighted_averaged',
#                      '/home4/tuannd/vbee-asr/asr-experiment/ckpt-mbf-vc-thu-am-aicc/checkpoint-100000',
#                      '/home4/tuannd/vbee-asr/asr-experiment/ckpt-mbf-vc-thu-am-aicc/checkpoint-100000',
#                      '/home4/tuannd/vbee-asr/asr-experiment/ckpt-mbf-vc-thu-am-aicc/checkpoint-100000',
#                     #  '/home4/tuannd/vbee-asr/asr-experiment/models/huggingface/mobiva_average_ckpt'
#                     ]
# params_dict = collections.OrderedDict()
# params_keys = None
# num_models = 0

# # for i, checkpoint in enumerate(checkpoint_wer.checkpoint.to_list()[:topk]):
# for i, checkpoint in enumerate(checkpoints_paths):
#     print(checkpoint)
#     num_models += 1
#     cpkt = Wav2Vec2ForCTC.from_pretrained(checkpoint)
#     model_params = cpkt.state_dict()
    
#     model_params_keys = list(model_params.keys())
#     if params_keys is None:
#         params_keys = model_params_keys
#     elif params_keys != model_params_keys:
#         raise KeyError(
#             "Expected list of params: {}, "
#             "but found: {}".format(params_keys, model_params_keys)
#         ) 
#     for k in params_keys:
#         p = model_params[k]
#         if isinstance(p, torch.HalfTensor):
#             p = p.float()
#         if k not in params_dict:
#             params_dict[k] = p.clone().to(dtype=torch.float64)
#             # NOTE: clone() is needed in case of p is a shared parameter
#         else:
#             params_dict[k] += p.to(dtype=torch.float64)

# print('num_models:', num_models)
# # num_models += 1

# final_state_dict = collections.OrderedDict()

# for k, v in params_dict.items():
#     v.div_(num_models)
#     # float32 overflow seems unlikely based on weights seen to date, but who knows
#     float32_info = torch.finfo(torch.float32)
#     for k, v in params_dict.items():
#         v = v.clamp(float32_info.min, float32_info.max)
#         final_state_dict[k] = v.to(dtype=torch.float32)

# float32_info = torch.finfo(torch.float32)

# config = Wav2Vec2Config.from_json_file(os.path.join(checkpoints_paths[-1], 'config.json'))

# average_model = Wav2Vec2ForCTC(config=config)
# average_model.load_state_dict(final_state_dict)


# average_model.save_pretrained('ckpt-mbf-vc-thu-am-aicc/average_model_v4')

checkpoints = [
    ('/home4/tuannd/vbee-asr/asr-experiment/models/huggingface/address_vc_weighted_averaged', 0.3),
    ('/home4/tuannd/vbee-asr/asr-experiment/ckpt-mbf-vc-thu-am-aicc/checkpoint-100000', 0.4),
    ('/home4/tuannd/vbee-asr/asr-experiment/ckpt-finetune-few-shot/checkpoint-442', 0.3),
]

checkpoint_dirs, weights = zip(*checkpoints)
# Ensure weights sum to 1
assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"

# Initialize dictionary to accumulate weighted parameters
params_dict = collections.OrderedDict()
params_keys = None

# Load each checkpoint and accumulate weighted parameters
for checkpoint_dir, weight in zip(checkpoint_dirs, weights):
    print(f"Loading checkpoint: {checkpoint_dir}")
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)
    state_dict = model.state_dict()
    
    if params_keys is None:
        params_keys = list(state_dict.keys())
        for k in params_keys:
            p = state_dict[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            params_dict[k] = weight * p.clone().to(dtype=torch.float64)
    else:
        if list(state_dict.keys()) != params_keys:
            raise ValueError(f"State dict keys do not match for {checkpoint_dir}")
        for k in params_keys:
            p = state_dict[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            params_dict[k] += weight * p.to(dtype=torch.float64)

# Load configuration from the first checkpoint
config = Wav2Vec2Config.from_pretrained(checkpoint_dirs[0])

# Create a new model with this configuration
average_model = Wav2Vec2ForCTC(config)

# Clamp and convert the averaged parameters to float32
float32_info = torch.finfo(torch.float32)
for k, v in params_dict.items():
    v = v.clamp(float32_info.min, float32_info.max).to(dtype=torch.float32)
    params_dict[k] = v

# Load the averaged state dictionary into the new model
average_model.load_state_dict(params_dict)

# Save the averaged model
save_dir = '/home4/tuannd/vbee-asr/asr-experiment/average_ckpt/average_model_v5'
average_model.save_pretrained(save_dir)

import subprocess
files_to_copy = [
    '/home4/tuannd/vbee-asr/asr-experiment/models/huggingface/vocab.json',
    '/home4/tuannd/vbee-asr/asr-experiment/models/huggingface/preprocessor_config.json'
]
for file in files_to_copy:
    subprocess.run(['cp', file, save_dir], check=True)
print(f"Averaged model saved to {save_dir}")
