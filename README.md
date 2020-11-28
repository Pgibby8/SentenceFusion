# SentenceFusion
## Installation
```
git clone https://github.com/Pgibby8/SentenceFusion.git
cd SentenceFusion
pip install .
```
## Summary
SentenceFusion leverages seq-2-seq language models to combine sentences in a semantic space,
allowing for the transfer of some components of one sentence to another.
This could have broad applications in helping write anything, say if you know what
you want a sentence to say, but don't like the exact way it looks or sounds, you can
fuse it with another sentence to help keep the creativity flowing
## Usage
1. download a seq-2-seq model that is trained to return the same sequence fed into it
along with its associated tokenized. We currently have prepared such a model that can
be accessed with `bert2bert = EncoderDecoderModel.from_pretrained('ralcanta/do_nothing_bert')`
and associated tokenizer `tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')`.
(TO DO: train a larger do-nothing model on more data)
2. Initialize a fuser
```
from SentenceFusion import sentence_fusion as sf

fuser = sf.SentenceFusion(tokenizer, bert2bert)
```
3. Fuse two sentences!
```
primary_sentence = "I have two cute dogs"
secondary_sentence = "to be or not to be, that is the question"
fuser.fuse(primary_sentence, secondary_sentence, delta=.5)
```
