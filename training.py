
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import LearningRateMonitor
#from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.optim as optim
import torch.nn as nn
from dataset import SanskritASRDataset,DataLoader
from tokenizer import tokenizer

train_dataset = SanskritASRDataset(split="train",tokenizer=tokenizer)
val_dataset= SanskritASRDataset (split="validation",tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=4,shuffle=True)

from model import Whisper, ModelDimensions,AudioEncoder,ResidualAttentionBlock,MultiHeadAttention,TextDecoder

dims = ModelDimensions(
    n_mels=128,  # Number of mel spectrogram channels
    n_audio_ctx=512,  # Context length for audio encoder
    n_audio_state=512,  # Audio encoder hidden state dimension
    n_audio_head=8,  # Number of audio encoder attention heads
    n_audio_layer=12,  # Number of audio encoder layers
    n_vocab=5000,  # Vocabulary size
    n_text_ctx=256,  # Context length for text decoder
    n_text_state=768,  # Text decoder hidden state dimension
    n_text_head=12,  # Number of text decoder attention heads
    n_text_layer=12,  # Number of text decoder layers
)


model=Whisper(dims=dims)
# Optional: Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()

for epoch in range(2):
    print(f"Epoch: {epoch+1}")
    for mel, text in train_dataloader:
        # Move data to device if needed
        print("Mel shape:", mel.shape)
        print("Text shape:", text.shape)
        mel = mel.to(device)
        text = text.to(device)

        # Forward pass
        logits = model(mel, text)

        # Calculate loss
        loss = loss_function(logits, text)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log losses and metrics (optional)
        print(f"Train loss: {loss.item()}")

# Save the trained model (optional)
torch.save(model.state_dict(), "trained_model.pth")