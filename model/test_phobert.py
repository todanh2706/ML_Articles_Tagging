import sys
import torch
from transformers import pipeline
from pyvi import ViTokenizer

# --- CONFIGURATION & SETUP ---

# 1. Force UTF-8 encoding for standard output
# This is crucial for Windows CMD/PowerShell to display Vietnamese characters correctly.
# On Mac/Linux, this is usually default, but good to have for cross-platform compatibility.
sys.stdout.reconfigure(encoding='utf-8')

# 2. Device Selection Logic
# - If CUDA is available (Windows with NVIDIA GPU), use it (device ID 0).
# - If Mac M1/M2 is detected, we force CPU (device ID -1) for this specific Inference task 
#   to avoid the 'LayerNorm' precision bug we found earlier.
# - Otherwise, use CPU.
device_id = -1 # Default to CPU
device_name = "CPU"

if torch.cuda.is_available():
    device_id = 0
    device_name = "CUDA (NVIDIA GPU)"
elif torch.backends.mps.is_available():
    device_id = -1 
    device_name = "CPU (Forced on Mac Apple Silicon for stability)"

print(f"--- SYSTEM INFO ---")
print(f"Running on device: {device_name}")

# --- HELPER FUNCTIONS ---

def get_prediction(text, model_name="vinai/phobert-base"):
    """
    Loads the model and predicts the masked word.
    """
    print(f"\nLoading model '{model_name}'... Please wait.")
    
    # Initialize the pipeline
    # 'fill-mask' is the task for predicting missing words
    nlp = pipeline("fill-mask", model=model_name, tokenizer=model_name, device=device_id)
    
    # Step 1: Tokenize (Segment) the input text using PyVi
    # PhoBERT requires words to be segmented (e.g., "Hà_Nội", "Việt_Nam")
    segmented_text = ViTokenizer.tokenize(text)
    
    # Step 2: Create the mask
    # We replace the target word with '<mask>' token
    # Note: PhoBERT uses '<mask>', while BERT uses '[MASK]'
    masked_text = segmented_text.replace("thủ_đô", "<mask>")
    
    print(f"Original text:  {text}")
    print(f"Segmented text: {segmented_text}")
    print(f"Masked input:   {masked_text}")
    print("-" * 40)
    
    # Step 3: Predict
    predictions = nlp(masked_text)
    
    return predictions

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # The input sentence
    input_text = "Hà Nội là thủ đô của Việt Nam."
    
    try:
        results = get_prediction(input_text)
        
        print(f"--- PREDICTION RESULTS ---")
        for i, item in enumerate(results):
            # Clean the token: remove underscores ('_') used for segmentation
            token_word = item['token_str'].replace("_", " ")
            confidence = item['score'] * 100
            
            print(f"Rank {i+1}: {token_word:<15} (Confidence: {confidence:.2f}%)")
            
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Tip: If you are on Windows, make sure you installed 'pyvi' and 'torch'.")