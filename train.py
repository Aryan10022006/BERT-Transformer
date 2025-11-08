"""
this is the complete training pipeline for BERT pre-training from scratch
it has two main parts:
1. main function to setup data loaders and start training
2. BertTrainer class encapsulating all training logic
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW

#importing from model.py and dataset.py
from model import BertForPreTraining, CONFIG, create_model
from dataset import BertPretrainingDataset

class BertTrainer:
    def __init__(self, model, learning_rate=5e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) 
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, train_loader, epoch_idx):
        self.model.train()
        total_loss = 0
        
        #looping through every batch
        for i, batch in enumerate(train_loader):
            #moving tensors to cpu/cuda
            input_ids = batch['input_ids'].to(self.device)
            segment_ids = batch['segment_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            mlm_labels = batch['mlm_labels'].to(self.device)
            nsp_label = batch['nsp_label'].to(self.device)
            
            #clearing old gradients
            self.optimizer.zero_grad()
            
            mlm_logits, nsp_logits = self.model(input_ids, segment_ids, attention_mask)
            #calculating losses
            mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, CONFIG['vocab_size']), mlm_labels.view(-1))
            nsp_loss = self.nsp_loss_fn(nsp_logits, nsp_label)
            
            loss = mlm_loss + nsp_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) 
            
            #applying fixes
            self.optimizer.step()
            
            total_loss += loss.item()
            
            #printing progress every 10 batches
            if i % 100 == 0: 
                print(f"  Epoch {epoch_idx} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Runs one full pass over the validation data.
        """
        #Setting model to evaluation mode
        self.model.eval()
        total_loss = 0
        total_nsp_correct = 0
        total_nsp_samples = 0

        #turning off the gradients
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                segment_ids = batch['segment_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                mlm_labels = batch['mlm_labels'].to(self.device)
                nsp_label = batch['nsp_label'].to(self.device)
                
                mlm_logits, nsp_logits = self.model(input_ids, segment_ids, attention_mask)
                
                # Calculate losses
                mlm_loss = self.mlm_loss_fn(mlm_logits.view(-1, CONFIG['vocab_size']), mlm_labels.view(-1))
                nsp_loss = self.nsp_loss_fn(nsp_logits, nsp_label)
                loss = mlm_loss + nsp_loss
                
                total_loss += loss.item()
                
                # Calculate NSP accuracy
                nsp_preds = torch.argmax(nsp_logits, dim=1)
                total_nsp_correct += (nsp_preds == nsp_label).sum().item()
                total_nsp_samples += nsp_label.size(0)
                
        avg_loss = total_loss / len(val_loader)
        nsp_accuracy = (total_nsp_correct / total_nsp_samples) * 100
            
        return avg_loss, nsp_accuracy

    def train(self, train_loader, val_loader, num_epochs):
        """
        The main "boss" function to orchestrate training and validation.
        """
        best_val_loss = float('inf')
        save_dir = "checkpoints"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            avg_train_loss = self.train_epoch(train_loader, epoch+1)
            avg_val_loss, nsp_accuracy = self.validate(val_loader)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            print(f"  Validation NSP Accuracy: {nsp_accuracy:.2f}%")
            
            #saving the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ New best model saved to {save_path}")

    def predict_masked_tokens(self, text, top_k=5):
        """
        Runs a demo of the trained model on a new sentence.
        """
        print(f"\nPredicting for: '{text}'")
        
        self.model.eval()
        tokens = self.tokenizer.tokenize(text)
        
        try:
            mask_index = tokens.index('[MASK]')
        except ValueError:
            print("Error: No [MASK] token found in text.")
            return

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        input_tensor = torch.tensor([input_ids]).to(self.device)
        seg_tensor = torch.zeros_like(input_tensor)
        mask_tensor = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            mlm_logits, _ = self.model(input_tensor, seg_tensor, mask_tensor)
            
        mask_logits = mlm_logits[0, mask_index + 1, :]
        top_k_indices = torch.topk(mask_logits, top_k).indices
        
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(top_k_indices)
        
        for i, token in enumerate(predicted_tokens):
            print(f"  {i+1}. {token}")

def main():
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 5e-5

    print("Initializing model...")
    model = create_model(CONFIG)
    
    #passing the model to the trainer
    trainer = BertTrainer(model, learning_rate=LEARNING_RATE)
    
    #loading the data
    print("Loading Train Dataset")
    train_data = BertPretrainingDataset(split='train', max_len=CONFIG['max_seq_len'])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Loading Validation Dataset...")
    val_data = BertPretrainingDataset(split='validation', max_len=CONFIG['max_seq_len'])
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # starting the training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS
    )

    # Running the demo
    print("\nTraining finished, running demo")
    trainer.predict_masked_tokens("The capital of France is [MASK].")
    trainer.predict_masked_tokens("The [MASK] is shining brightly today.")
    trainer.predict_masked_tokens("He went to the [MASK] to buy some milk.")

if __name__ == "__main__":
    main()