import os
import re
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Load the model and tokenizer
model_path = Path(os.path.join(PROJECT_DIR, "models", "gpt2", "model"))
tokenizer_path = Path(os.path.join(PROJECT_DIR, "models", "gpt2", "tokenizer"))

model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_lyrics(genre, prompt, min_length=200, max_length=600):
    genre_context = f"[s:genre]{genre}[e:genre]"
    lyrics_context = f"[s:lyrics]"
    full_prompt = f"{genre_context}{lyrics_context} {prompt}"
    
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

    eos_token_id = tokenizer.encode('[e:lyrics]', add_special_tokens=False)[0]

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.8
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    clean_text = generated_text.split('[e:lyrics]')[0]  # Strip everything after end lyrics tag
    clean_text = re.sub(r'\[s:genre\].*?\[e:genre\]', '', clean_text)  # Remove genre tags
    clean_text = clean_text.replace(lyrics_context, "").strip()  # Remove lyrics start tag

    return clean_text

genres = ["Pop", "Rap"]
prompt = False
initial_prompt = ''
for genre in genres:
    if prompt:
        initial_prompt = input(f'Enter the first few lines for {genre} generation, or leave empty for the model to decide:\n')
    generated_lyrics = generate_lyrics(genre, initial_prompt)
    clean_text = re.sub(r'\[s:genre\]', '', generated_lyrics) # Remove [s:genre] tags
    clean_text = re.sub(r'\<|endoftext|>', '', clean_text)
    clean_text = re.sub(r'\[sgenre\]', '', clean_text) 
    clean_text = re.sub(r'\[egenre\]', '', clean_text) 
    clean_text = re.sub(r'\[e:genre\]', '', clean_text) 
    clean_text = re.sub(r'\||', '', clean_text)
    clean_text = re.sub(r':', '', clean_text)

    print(f"Generated Lyrics for {genre}:\n\n",clean_text + '\n')