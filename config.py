# pretrained_model = "/home3/thanhpv/pretrained_wav2vec/pretrain-aicc"
pretrained_model = "nguyenvulebinh/wav2vec2-base-vi"

# pretrained_model = (
#     "/home3/khanhnd/multitask/checkpoint-86000"
# )
name="vin_bigdata"
training_data = f"/home4/khanhnd/improved_unimatch/data_csv/{name}_sample.csv"
# training_data = '/home4/tuannd/vbee-asr/asr-data/train_b2_sampled_trimmed.BEST'
vocab_file = "/home4/khanhnd/thesis/asr_base/vocab.json"
checkpoint_dir = f"/home3/khanhnd/{name}_sample_1"
# training args
num_epochs = 10
batch_size = 6
cache_dir=f"/home3/khanhnd/cache/{name}"
learning_rate = 2e-5
print("Siuuuu")
