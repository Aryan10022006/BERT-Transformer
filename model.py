#building bert from scratch.
#plan: build small parts (embeddings, attention, ffn),
#then combine them into the final model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#it has all the same parts as the real one, just smaller
CONFIG = {
    'vocab_size': 30522,
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'intermediate_size': 512, # 4 * hidden_size
    'max_seq_len': 128,
    'type_vocab_size': 2, #for segment embeddings (sent A vs sent B)
    'dropout_prob': 0.1,
    'layer_norm_eps': 1e-12,
}


#the embedding layer
#sums token + position + segment embeddings


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.position_embeddings = nn.Embedding(config['max_seq_len'], config['hidden_size'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['hidden_size'])
        
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_prob'])
        
    def forward(self, input_ids, token_type_ids):
        batch_size, seq_len = input_ids.size()
        
        #create position_ids on the fly (0, 1, 2, ...)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        #adding all up
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


#2. multi-head self-attention
#this is the core 'engine' of the transformer


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.hidden_size = config['hidden_size']
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config['dropout_prob'])

    #helper to split hidden_size into (num_heads, head_dim)
    def _split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # (batch, head, seq, dim)
    
    #helper to merge heads back
    def _combine_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.hidden_size)
        
    def forward(self, hidden_states, attention_mask):
        batch_size = hidden_states.size(0)
        
        #1. get q, k, v
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        #2. splitting heads
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)
        
        #3. getting attention scores (q @ k.T)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim) #scale
        
        #4. applying mask (so we don't look at padding tokens)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(mask == 0, -10000.0) #set padding to -infinity
        
        #5. softmax
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        #6. get context (probs @ v)
        context = torch.matmul(probs, v)
        
        #7. combine heads
        context = self._combine_heads(context, batch_size)
        
        #8. final linear layer
        output = self.dense(context)
        return output


#feed-forward network
#just a simple processor that runs after attention
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.dense_2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout_prob'])
        
    def forward(self, hidden_states):
        x = self.dense_1(hidden_states)
        x = F.gelu(x) #gelu is the activation bert uses
        x = self.dropout(x)
        x = self.dense_2(x)
        return x


#encoder block (one full layer)
#combines attention + ffn with layernorm and residual (skip) connections

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.ffn_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['dropout_prob'])
        
    def forward(self, hidden_states, attention_mask):
        
        #attention part
        normed_input = self.attention_layer_norm(hidden_states)
        attn_output = self.attention(normed_input, attention_mask)
        attn_output = self.dropout(attn_output)
        hidden_states = hidden_states + attn_output #residual
        
        #ffn part
        normed_input = self.ffn_layer_norm(hidden_states)
        ffn_output = self.feed_forward(normed_input)
        ffn_output = self.dropout(ffn_output)
        output = hidden_states + ffn_output #residual
        
        return output


#bert model (the backbone)
#this is just the embeddings + a stack of the encoder blocks
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(config) for _ in range(config['num_layers'])
        ])
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.embeddings(input_ids, token_type_ids)
        for block in self.encoder_blocks:
            x = block(x, attention_mask)
        return x


#the pre-training heads
#these sit on top of the bert model for the two pre-training tasks

#mlm head: predicts masked words
class MlmHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.decoder = nn.Linear(config['hidden_size'], config['vocab_size'])

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x) #output is (batch, seq, vocab_size)
        return x

#nsp head: predicts if sentence b follows a
class NspHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config['hidden_size'], 2)
        
    def forward(self, pooled_output):
        #only takes the [cls] token's output
        return self.seq_relationship(pooled_output)


#complete model
#combines the backbone with the two heads
class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.mlm_head = MlmHead(config)
        self.nsp_head = NspHead(config)
        self.config = config #so train.py can see the vocab_size
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        #1. get the final hidden states from the backbone
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask)
        
        #2. get mlm scores for all tokens
        mlm_scores = self.mlm_head(sequence_output)
        
        #3. get nsp score from *only* the [cls] token (at index 0)
        pooled_output = sequence_output[:, 0]
        nsp_scores = self.nsp_head(pooled_output)
        
        return mlm_scores, nsp_scores


#helper to create the model and init weights properly
def create_model(config):
    print("making a new bert model...")
    model = BertForPreTraining(config) 
    
    #this is the standard bert weight initialization
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    model.apply(_init_weights)
    print("...model created and weights initialized.")
    return model


if __name__ == "__main__":
    model = create_model(CONFIG)
    print(f"model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    fake_ids = torch.randint(0, CONFIG['vocab_size'], (4, 16))
    fake_segments = torch.zeros(4, 16, dtype=torch.long)
    fake_mask = torch.ones(4, 16, dtype=torch.long)
    
    with torch.no_grad():
        mlm_scores, nsp_scores = model(fake_ids, fake_segments, fake_mask)
        
    print(f"mlm output shape: {mlm_scores.shape}") #should be (4, 16, 30522)
    print(f"nsp output shape: {nsp_scores.shape}") #should be (4, 2)
    print("shapes look good. test passed.")