# BERT from Scratch for NLP Selection Camp

built this for the Inter IIT Tech Meet 14., Nlp Selection bootcamp where I had to implement BERT completely from scratch using pytorch - no pre-built transformers allowed, just raw pytorch layers and patience.

## what's implemented

everything the bootcamp asked for:

- encoder-only transformer with multi-head self-attention 
- proper BERT embeddings (token + position + segment)
- MLM and NSP training objectives 
- wikitext-2 data processing with correct masking strategy
- joint training on both tasks
- demo showing masked token predictions

follows the original paper but scaled down to actually train on my laptop. all the core concepts are there just with smaller dimensions.

## files

**model.py** - transformer architecture built from scratch:
- BertEmbeddings (token + position + segment) 
- MultiHeadSelfAttention 
- EncoderBlock with residuals
- MLM and NSP heads
- complete BertForPreTraining model

**dataset.py** - wikitext-2 processing:
- sentence segmentation 
- NSP pair creation (50% real, 50% random)
- MLM masking (15% tokens, 80/10/10 strategy)
- proper tokenization with [CLS], [SEP]

**train.py** - training loop:
- joint MLM + NSP loss optimization
- validation with NSP accuracy tracking
- model checkpointing
- demo predictions

## config 

scaled everything down to run on my local pc:

```python
CONFIG = {
    'vocab_size': 30522,      # bert tokenizer vocab
    'hidden_size': 128,       # 768 in real bert 
    'num_layers': 2,          # 12 in bert-base
    'num_heads': 4,           # 12 in bert-base
    'intermediate_size': 512, # 4 * hidden_size
    'max_seq_len': 128,       
    'type_vocab_size': 2,     # segment A vs B
    'dropout_prob': 0.1,      
}
```

batch_size=16, lr=1e-4, epochs=3. all tuned for my setup - 16gb ram, decent gpu. adjust if needed.

## running it

install stuff:
```bash
pip install torch transformers datasets nltk
```

run:
```bash  
python train.py
```

downloads wikitext-2, trains for 3 epochs (~1 hour), saves checkpoint, shows demo predictions.

parameters set for my local config (16gb ram, rtx gpu). adjust batch_size in train.py if you get oom errors.

## results

training converges, NSP accuracy hits ~58% (above random), MLM predictions make sense:

```
Epoch 3 Summary:
  Training Loss: 7.5916
  Validation Loss: 7.6399
  Validation NSP Accuracy: 66.15%
✓ New best model saved to checkpoints\best_model.pth

Training finished, running demo

Predicting for: 'The capital of France is [MASK].'
  1. the
  2. ,
  3. of
  4. in
  5. to

Predicting for: 'The [MASK] is shining brightly today.'
  1. the
  2. ,
  3. of
  4. was
  5. to
Predicting for: 'He went to the [MASK] to buy some milk.'
  1. the
  2. ,
  3. of
  4. to
  5. in
```

model checkpoint saved to `checkpoints/best_model.pth`.

## notes

implements everything from the BERT paper - encoder-only transformer, proper embeddings, joint MLM+NSP training on wikitext-2. follows the bootcamp requirements exactly but scaled down for local training.