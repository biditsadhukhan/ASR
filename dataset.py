import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np 
my_tokenizer = AutoTokenizer.from_pretrained("Bidwill/whisper-medium-sanskrit-out-domain")

from torch.utils.data import Dataset
from torchaudio import load,functional
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


class SanskritASRDataset(Dataset):
    def __init__(self,  split,tokenizer,data_dir="/home/bidit/term_project/Vāksañcayaḥ- Sanskrit_ASR_Corpus"):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer=tokenizer

        # Load the dataset from the file structure
        self.data = []
        split_dir = os.path.join(data_dir, split)
        for speaker_folder in os.listdir(split_dir):
            speaker_path = os.path.join(split_dir, speaker_folder)

            if os.path.isdir(speaker_path):
                transcript_file = f"{speaker_folder}.txt"
                transcript_path = os.path.join(speaker_path, transcript_file)

                if os.path.exists(transcript_path):
                    with open(transcript_path, "r", encoding="utf-8") as transcript_file:
                        for line in transcript_file:
                            line = line.strip().split('\t')
                            if len(line) == 2:
                                audio_name = line[0]
                                transcript = line[1]
                                audio_path = os.path.join(speaker_path, audio_name + ".wav")
                                speaker_id = int(speaker_folder[2:])

                                if os.path.exists(audio_path):
                                    self.data.append({"audio": audio_path, "transcription": transcript, "speaker_id": speaker_id})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, transcript, speaker_id = self.data[idx]["audio"], self.data[idx]["transcription"], self.data[idx]["speaker_id"]
        waveform, sample_rate = load(audio_path, normalize=True)
       
    # Load audio and extract features
        waveform, sample_rate = load(audio_path, normalize=True)
        if sample_rate != 16000:
            waveform = functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.squeeze().unsqueeze(0).float()
        mel_specgram = MelSpectrogram(sample_rate=16000)(waveform)
        mel_spectogram=pad_or_trim(mel_specgram)

    # Tokenize transcript using the transformer tokenizer
        with self.tokenizer.as_target_tokenizer():
            encoded_tokens = self.tokenizer(transcript, return_tensors="pt")
            input_ids = encoded_tokens.input_ids
            attention_mask = encoded_tokens.attention_mask

    # Return audio features, transcript tokens, attention masks, and other data
        return mel_spectogram, input_ids,attention_mask





train_dataset = SanskritASRDataset(split="train",tokenizer=my_tokenizer)
val_dataset= SanskritASRDataset (split="validation",tokenizer=my_tokenizer)
oov_dataset=SanskritASRDataset(split="oov",tokenizer=my_tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=4,shuffle=True)
print(train_dataset[0])
#print(val_dataset)
# print(len(train_dataset))
# print(len(val_dataset))
# oov_dataloader = DataLoader(oov_dataset, batch_size=4)
# print(oov_dataset[0])
for mel, text,attention in train_dataloader:
    # Print shapes for all three elements
    print("Mel shape:", mel.shape)
    print("Text shape:", text.shape)
    print("Text shape:",attention.shape)
    

   
