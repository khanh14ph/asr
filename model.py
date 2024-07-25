import torch.nn as nn
import torch     
from transformers import Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
import torch.nn.functional as F
_HIDDEN_STATES_START_POSITION = 2
def scheduling_func(e, E=200, t=0.3):
    return min(max((e-1)/(E-1), t), 1-t)
class Wav2Vec2ForCTCDistil(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head_t = nn.Linear(config.output_hidden_size, config.vocab_size)
        self.lm_head_s = nn.Linear(config.output_hidden_size, config.vocab_size)
        self.s_layer=8
        self.step=0
    def forward(
        self,
        input_values,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states= True,
        return_dict= None,
        labels= None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        temperature = 1
        hidden_states = outputs.hidden_states
        s_logits = self.lm_head_s(hidden_states[self.s_layer])
        t_logits = self.lm_head_t(hidden_states[-1])
        # hidden_states = self.dropout(hidden_states)
        l_skd = F.kl_div(F.log_softmax(t_logits/temperature, dim=2), F.softmax(s_logits/temperature, dim=2))
        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: {self.config.vocab_size}"
                )

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            s_log_probs = nn.functional.log_softmax(
                s_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            t_log_probs = nn.functional.log_softmax(
                t_logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                s_loss = nn.functional.ctc_loss(
                    s_log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                t_loss = nn.functional.ctc_loss(
                    t_log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
        alpha = scheduling_func(e=self.step+1, E=500000, t=0.3)
        loss = (1-alpha)*t_loss + alpha*(s_loss + l_skd)
        self.step+=1
        
        if not return_dict:
            output = (t_logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=t_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
