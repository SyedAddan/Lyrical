from datasets import Dataset
from transformers import GPT2TokenizerFast
import torch
import os
import re
import numpy as np
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import AdamW, get_scheduler, EarlyStoppingCallback

os.environ['WANDB_DISABLED'] = 'true'

if os.path.exists('/kaggle/input/parameters/cleaned_lyrics_data.txt'):
    file_path = '/kaggle/input/parameters/cleaned_lyrics_data.txt'
else:
    file_path = '/kaggle/working/cleaned_lyrics_data2.txt'

from transformers import TrainerCallback

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step: {state.global_step}, Training Loss: {logs.get('loss', 'N/A')}, Validation Loss: {logs.get('eval_loss', 'N/A')}")

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.epochs = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Save evaluation loss after each evaluation
        if metrics and 'eval_loss' in metrics:
            self.eval_losses.append(metrics['eval_loss'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Save training loss after each logging step
        if logs and 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.epochs.append(state.epoch)  # Save the current epoch

    def get_metrics(self):
        return self.epochs, self.train_losses, self.eval_losses

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split the content into lyric blocks
        self.lyric_blocks = re.findall(r'\[s:genre\].*?\[e:genre\]\[s:lyrics\](.*?)\[e:lyrics\]', content, re.DOTALL)
        if not self.lyric_blocks:
            raise ValueError("No lyric blocks found. Check the format of the input file.")

        self.examples = []
        for block in self.lyric_blocks:
            bpe_tokens = tokenizer(block.strip(), truncation=True, max_length=block_size, padding="max_length", return_tensors="pt")
            self.examples.append({
                'input_ids': bpe_tokens['input_ids'][0],
                'attention_mask': bpe_tokens['attention_mask'][0],
                'labels': bpe_tokens['input_ids'][0]  
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
    
# Load tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
special_tokens_dict = {'additional_special_tokens': ['[s:genre]', '[e:genre]', '[s:lyrics]', '[e:lyrics]']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token

# Create dataset
block_size = 128 # Adjust based on your GPU memory
dataset = LyricsDataset(tokenizer, file_path, block_size)

from sklearn.model_selection import train_test_split

train_dataset, eval_dataset = train_test_split(dataset, test_size=0.1)  # 10% for validation

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch", 
    save_strategy="epoch", 
    logging_strategy="epoch",              
    num_train_epochs=40,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    greater_is_better=False
)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Initialize the learning rate scheduler
num_training_steps = len(train_dataset) * training_args.num_train_epochs
lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps,
)

metrics_logger = MetricsLoggerCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[metrics_logger, EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Save the model and configuration
model.save_pretrained('/kaggle/working/results/best/')

tokenizer.save_pretrained('./results/tokenizer/')

import json

# Save TrainingArguments to a JSON file
args_dict = training_args.to_dict()  # Convert TrainingArguments to a dictionary
with open('/kaggle/working/results/training_args.json', 'w') as f:
    json.dump(args_dict, f, indent=4)

    
import matplotlib.pyplot as plt

# Retrieve logged metrics
epochs, train_losses, eval_losses = metrics_logger.get_metrics()

# Create the plot
if epochs:
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, eval_losses, label='Validation Loss', marker='x', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over All Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
    if len(epochs) > 20:  # Check if there are at least 20 epochs
        zoom_range_start = -20  # Last 20 epochs
        plt.figure(figsize=(12, 6))
        plt.plot(epochs[zoom_range_start:], train_losses[zoom_range_start:], label='Training Loss', marker='o', color='blue')
        plt.plot(epochs[zoom_range_start:], eval_losses[zoom_range_start:], label='Validation Loss', marker='x', color='orange')
        plt.xlabel('Epochs (Last 20)')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Last 20 Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
    
else:
    print("No data to plot. Check the data collection process.")