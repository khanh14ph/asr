from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
import torch
device="cuda"
class SimpleWav2Vec(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(
            config,
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values,attention_mask=None, 
                            output_hidden_states=True,).hidden_states[-1]
        return outputs
model=SimpleWav2Vec.from_pretrained("nguyenvulebinh/wav2vec2-base-vi").to(device)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("nguyenvulebinh/wav2vec2-base-vi")
import librosa
speech, sr=librosa.load("/home4/khanhnd/vivos/test/waves/VIVOSDEV01/VIVOSDEV01_R002.wav")
# print("speech",speech.shape)
speech=[speech for i in range(10)]
acoustic = feature_extractor(speech, sampling_rate = 16000)
acoustic = torch.tensor(acoustic.input_values, device=device)

logits=model(acoustic)
print(logits.shape)
