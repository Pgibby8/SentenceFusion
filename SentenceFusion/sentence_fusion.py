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
        """
        initializes a sentence fusion object
        :param tokenizer: The tokenizer to be used to tokenize input sentences
        :type tokenizer: a tokenizer from transformers (the same one associated with the encoder decoder)
        :param seq2seq: an encoder decoder model trained to output exactly what is input into it
        :type seq2seq: a encoder decoder model from transformers
        :param decoder_start_token_id: the id of the start sentence token expected by the encoder decoder model
        :type decoder_start_token_id: int, shouldn't need to be changed
        :param eos_token_id: the id of the end of sentence token expected by the encoder decoder model
        :type eos_token_id: int, shoudln't need to be changed
        """
        self.tokenizer = tokenizer
        self.seq2seq = seq2seq
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

    def fuse(self, primary_sent, secondary_sent, num_beams=3, delta=1, mode='mean',
             rep_penalty=10.0, no_ngram_repeats=2):
        """
        Fuses two sentences, taking the semantics from the first sentence and higher level characteristics from the second
        :param primary_sent: the sentence from which to extract the semantics
        :type primary_sent: string
        :param secondary_sent: the sentence from which to extract high level features
        :type secondary_sent: string
        :param num_beams: the number of beams in the beam search at the end
        :type num_beams: int > 0
        :param delta: the 'strength' of the transfer. Lower values of delta maintain more of the semantics,
                      higher value of delta increase the effect of the secondary sentence
        :type delta: float > 0
        :param mode: the method of transfer.
        :type mode: a string from ['mean', 'align']
        """
        encoder_content_ids = self.tokenizer(primary_sent, return_tensors='pt').input_ids
        encoder_style_ids = self.tokenizer(secondary_sent, return_tensors='pt').input_ids

        # define decoder start token ids
        input_ids = torch.ones((num_beams, 1), device=self.seq2seq.device, dtype=torch.long)
        input_ids = input_ids * self.decoder_start_token_id
        # add encoder_outputs to model keyword arguments
        encoder_content_outputs = self.seq2seq.get_encoder()(encoder_content_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        encoder_style_outputs = self.seq2seq.get_encoder()(encoder_style_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
        # this is where we try to apply the sentence fusion
        if mode == 'mean':
            style = torch.mean(encoder_style_outputs[0], axis=1).unsqueeze(1)
            content = encoder_content_outputs.last_hidden_state
            difference = style - content
            style.shape, content.shape, difference.shape
            style_direction = difference / torch.norm(difference, dim=2).unsqueeze(2)
            encoder_content_outputs.last_hidden_state += delta * style_direction
        elif mode == 'align':
            n = encoder_content_outputs[0].shape[1]
            if encoder_style_outputs[0].shape[1] < n:
                raise ValueError('For the align method, the secondary sentence must be the same length or longer than the primary')
            encoder_content_outputs.last_hidden_state += delta * encoder_style_outputs[0][:,:n,:]
        else:
            raise ValueError(f'{mode} is not a recognized mode')
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
            RepetitionPenaltyLogitsProcessor(rep_penalty),
            NoRepeatNGramLogitsProcessor(no_ngram_repeats)
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

    def fuse_interactive(self, secondary_sent, num_beams=3, delta=1, mode='mean',
             rep_penalty=10.0, no_ngram_repeats=2):
        call_fuse = lambda so_far: self.fuse(so_far, secondary_sent, num_beams, delta, mode, rep_penalty, no_ngram_repeats)
        curr_sentence = ""
        first = True
        while True:
            msg_prompt = "Continue: " if not first else "Enter a sentence to begin with: "
            new_message = input(msg_prompt)
            if new_message == "":
                break
            if not first: 
                curr_sentence += " " + new_message
            else:
                curr_sentence = new_message
            first = False
            curr_sentence = call_fuse(curr_sentence)
            print("Story so far:", curr_sentence)
        print("\n Final story:")
        print("------------------")
        print(curr_sentence)
        print()
        return curr_sentence



