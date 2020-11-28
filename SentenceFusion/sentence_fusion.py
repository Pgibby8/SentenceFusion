import torch
import numpy as np
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    BeamSearchScorer,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)


class SentenceFusion():
    def __init__(self, tokenizer, seq2seq, decoder_start_token_id=101, eos_token_id=102):
        self.tokenizer = tokenizer
        self.seq2seq = seq2seq
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

    def fuse(self, primary_sent, secondary_sent, num_beams=3, delta=1):
        encoder_content_ids = self.tokenizer(primary_sent, return_tensors='pt').input_ids
        encoder_style_ids = self.tokenizer(secondary_sent, return_tensors='pt').input_ids

        # define decoder start token ids
        input_ids = torch.ones((num_beams, 1), device=self.seq2seq.device, dtype=torch.long)
        input_ids = input_ids * self.decoder_start_token_id
        # add encoder_outputs to model keyword arguments
        encoder_content_outputs = self.seq2seq.get_encoder()(encoder_content_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        encoder_style_outputs = self.seq2seq.get_encoder()(encoder_style_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        # this is where we try to apply the sentence fusion
        style = torch.mean(encoder_style_outputs[0], axis=1).unsqueeze(1)
        content = encoder_content_outputs.last_hidden_state
        difference = style - content
        style.shape, content.shape, difference.shape
        style_direction = difference / torch.norm(difference, dim=2).unsqueeze(2)
        encoder_content_outputs.last_hidden_state += delta * style_direction
        model_kwargs = {
            "encoder_outputs": encoder_content_outputs
        }
        # instantiate beam scorer
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            max_length=self.seq2seq.config.max_length,
            num_beams=num_beams,
            device=self.seq2seq.device,
        )
        # instantiate logits processors
        logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(5, eos_token_id=self.eos_token_id),
            RepetitionPenaltyLogitsProcessor(10.0),
            NoRepeatNGramLogitsProcessor(2)
        ])
        # instantiate logits processors
        logits_warper = LogitsProcessorList([
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7),
        ])
        outputs = self.seq2seq.beam_sample(
            input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
