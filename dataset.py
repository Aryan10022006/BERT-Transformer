"""
this file creates pytorch dataset for BERT pretraining with MLM and NSP tasks which transforms raw text of wikitext-2
into the exact format BERT needs for pretraining.
BERT is trained on two tasks simultaneously:

Masked Language Modeling - predicting masked words
Next Sentence Prediction - determining if two sentences follow each other

The dataset will output batches containing:
- input_ids: Tokenized input sequences
- segment_ids: Sentence A vs B indicators
- attention_mask: Padding mask
- mlm_labels: Labels for masked language modeling 
- nsp_label: Binary label for next sentence prediction

this file output the tensors required for both MLM and NSP tasks during BERT pretraining.
"""

#IMPORTING LIBRARIES
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import nltk
import random
import numpy as np
from typing import List, Tuple, Dict

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BertPretrainingDataset(Dataset):
    def __init__(self, split: str, max_len: int, tokenizer_name: str = 'bert-base-uncased', mlm_probability: float = 0.15):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_len
        self.mlm_probability = mlm_probability
        self.dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        self.documents = self._process_documents(self.dataset)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        sentence_a, sentence_b, nsp_label = self._create_nsp_example(idx)
        input_ids, segment_ids = self._tokenize_and_format(sentence_a, sentence_b)
        input_ids, mlm_labels = self._apply_mlm_masking(input_ids)

        attention_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * padding_length
        segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        mlm_labels += [-100] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long),
            'nsp_label': torch.tensor(nsp_label, dtype=torch.long)
        }
    
    def _process_documents(self, dataset):
        all_documents = []
        for article in dataset:
            text = article['text'].strip()
            if not text or text.startswith(' = '):
                continue
            
            doc_sentences = nltk.sent_tokenize(text)
            if doc_sentences:
                all_documents.append(doc_sentences)
        return all_documents

    def _create_nsp_example(self, idx):
        doc = self.documents[idx]
        
        if len(doc) < 2:
            return self._create_nsp_example(random.randint(0, len(self) - 1))

        sent_a_idx = random.randint(0, len(doc) - 2)
        sent_a = doc[sent_a_idx]
        
        if random.random() > 0.5:
            sent_b = doc[sent_a_idx + 1]
            nsp_label = 0
        else:
            random_doc_idx = random.randint(0, len(self) - 1)
            while random_doc_idx == idx:
                random_doc_idx = random.randint(0, len(self) - 1)
            
            other_doc = self.documents[random_doc_idx]
            sent_b = other_doc[random.randint(0, len(other_doc) - 1)]
            nsp_label = 1
            
        return sent_a, sent_b, nsp_label
    
    def _tokenize_and_format(self, sentence_a, sentence_b):
        tokens_a = self.tokenizer.tokenize(sentence_a)
        tokens_b = self.tokenizer.tokenize(sentence_b)

        max_tokens_for_each = (self.max_seq_length - 3) // 2
        
        tokens_a = tokens_a[:max_tokens_for_each]
        tokens_b = tokens_b[:max_tokens_for_each]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token] + \
                 tokens_b + [self.tokenizer.sep_token]
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        return input_ids, segment_ids
    
    def _apply_mlm_masking(self, input_ids):
        masked_ids = input_ids.copy()
        labels = [-100] * len(input_ids)
        
        special_token_ids = {self.tokenizer.cls_token_id, 
                             self.tokenizer.sep_token_id, 
                             self.tokenizer.pad_token_id}
        
        candidates = [i for i, token_id in enumerate(input_ids) if token_id not in special_token_ids]
        
        num_to_mask = max(1, int(len(candidates) * self.mlm_probability))
        if len(candidates) == 0:
            num_to_mask = 0
            
        masked_indices = random.sample(candidates, num_to_mask)
        
        for idx in masked_indices:
            labels[idx] = masked_ids[idx]

            prob = random.random()

            if prob < 0.8:
                masked_ids[idx] = self.tokenizer.mask_token_id
            elif prob < 0.9:
                masked_ids[idx] = random.randint(0, self.tokenizer.vocab_size - 1)
            
        return masked_ids, labels
    
if __name__ == "__main__":
    print("Testing the class made: ")
    
    dataset = BertPretrainingDataset(split='train', max_len=128)
    
    print(f"Total documents processed: {len(dataset)}")
    
    item = dataset[0]
    
    print("item keys and their shapes:")
    for key, value in item.items():
        print(f"{key}: {value.shape}")
        
    print("decoding first example:")
    print(f"NSP Label: {item['nsp_label']} (0 = IsNext, 1 = NotNext)")
    
    input_tokens = dataset.tokenizer.convert_ids_to_tokens(item['input_ids'])
    print("Input:", " ".join(input_tokens))

    mlm_labels = item['mlm_labels']
    for i, token in enumerate(input_tokens):
        if mlm_labels[i] != -100:
            original_token = dataset.tokenizer.convert_ids_to_tokens([mlm_labels[i].item()])
            print(f"  > Masked '{token}' at index {i}, original was '{original_token[0]}'")

    print("\ntesting dataLoader...")
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    
    print("batch shapes:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
        
    print("\nTest complete!")