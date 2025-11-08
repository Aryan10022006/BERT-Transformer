import torch
import torch.nn.functional as F
from transformers import BertTokenizer

#gotta import our own model files
from model import BertForPreTraining, CONFIG

MODEL_PATH = "checkpoints/best_model.pth" #path to my saved model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5
def load_my_model(model_path, config, device):
    print("loading my trained model...")
    
    #1. make the blank model
    model = BertForPreTraining(config)
    
    #2. load the saved weights into it
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #3. put it on the gpu/cpu
    model.to(device)
    
    #4. set it to eval mode (super important, turns off dropout)
    model.eval()
    
    print("...model's ready!")
    return model

def predict(text, model, tokenizer, device, top_k):
    print(f"\npredicting for: '{text}'")
    
    #1. tokenize
    tokens = tokenizer.tokenize(text)
    
    #2. find the mask
    try:
        mask_index = tokens.index('[MASK]')
    except ValueError:
        print("whoops, no [MASK] token in that sentence.")
        return

    #3. add the [cls] and [sep] tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    #4. make all the tensors
    input_tensor = torch.tensor([input_ids]).to(device)
    seg_tensor = torch.zeros_like(input_tensor)
    mask_tensor = torch.ones_like(input_tensor)
    
    #5. run the model (no gradients needed)
    with torch.no_grad():
        mlm_logits, _ = model(input_tensor, seg_tensor, mask_tensor)
        
    #6. get the scores for *just* the masked word
    #it's +1 because of the [CLS] at the start
    mask_logits = mlm_logits[0, mask_index + 1, :]
    
    #7. get the top 5
    top_k_indices = torch.topk(mask_logits, top_k).indices
    
    #8. turn the IDs back into words
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indices)
    
    #9. print 'em
    for i, token in enumerate(predicted_tokens):
        print(f"  {i+1}. {token}")

def main():
    #load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    #load my trained model
    model = load_my_model(MODEL_PATH, CONFIG, DEVICE)
    
    print("\nmodel's loaded. type a sentence with [MASK].")
    print("type 'exit' to quit.")
    
    #this just keeps asking for input
    while True:
        text = input("> ")
        
        if text.lower() == 'exit':
            break
        
        predict(text, model, tokenizer, DEVICE, TOP_K)

if __name__ == "__main__":
    main()