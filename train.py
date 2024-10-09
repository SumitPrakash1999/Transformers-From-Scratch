import sentencepiece as spm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from utils import TranslationDataset, create_data_loader, calculate_bleu, epoch_time
from encoder import Encoder
from decoder import Decoder
from transformer import Transformer

# Function to save sentences to a temporary file
def save_sentences_to_file(sentences, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

# Load the datasets from files
def load_data(file_path_src, file_path_trg):
    with open(file_path_src, 'r', encoding='utf-8') as f_src, open(file_path_trg, 'r', encoding='utf-8') as f_trg:
        src_sentences = [line.strip() for line in f_src.readlines()]
        trg_sentences = [line.strip() for line in f_trg.readlines()]
    return src_sentences, trg_sentences

# Load the dataset
train_src, train_trg = load_data('train.en', 'train.fr')
val_src, val_trg = load_data('dev.en', 'dev.fr')

# Save the sentences to temporary files for SentencePiece model training
save_sentences_to_file(train_src, 'train_src.txt')
save_sentences_to_file(train_trg, 'train_trg.txt')

# Set the vocabulary size
vocab_size = 10000

# Train SentencePiece models to tokenize the source and target language
spm.SentencePieceTrainer.train(f'--input=train_src.txt --model_prefix=eng_model --user_defined_symbols=<pad> --vocab_size={vocab_size}')
spm.SentencePieceTrainer.train(f'--input=train_trg.txt --model_prefix=fr_model --user_defined_symbols=<pad> --vocab_size={vocab_size}')

# Load the trained SentencePiece models
en_sp = spm.SentencePieceProcessor()
en_sp.load('eng_model.model')

fr_sp = spm.SentencePieceProcessor()
fr_sp.load('fr_model.model')

# Function to initialize the transformer model
def initialize_model(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device, config):
    encoder = Encoder(config)
    decoder = Decoder(config)
    model = Transformer(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    return model, optimizer, criterion

# Training loop
def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for _, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        trg_input = trg[:, :-1]
        trg_pad_mask = (trg_input != 0).unsqueeze(1).unsqueeze(2)
        output = model(src, src_mask, trg_input, trg_pad_mask)
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

# Train and evaluate function
def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, n_epochs, clip, device, trg_vocab, trg_pad_idx):
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip, device)
        val_loss, predictions, targets = evaluate(model, val_loader, criterion, device)
        val_bleu = calculate_bleu(predictions, targets, trg_vocab, trg_pad_idx)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {val_loss:.3f} | Val BLEU: {val_bleu:.2f}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'transformer.pt')

# Load the dataset and tokenizers, initialize the model, and train it
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model configurations
    config = {
        'encoder_vocab_size': 10000,
        'decoder_vocab_size': 10000,
        'd_embed': 512,
        'N_encoder': 6,
        'N_decoder': 6,
        'd_ff': 2048,
        'h': 8,
        'dropout': 0.1,
        'max_seq_len': 100
    }
    
    # Load datasets and tokenizers
    train_loader = create_data_loader(train_src, train_trg, en_sp.encode_as_ids, fr_sp.encode_as_ids, 128)
    val_loader = create_data_loader(val_src, val_trg, en_sp.encode_as_ids, fr_sp.encode_as_ids, 128)

    # Initialize and train the model
    model, optimizer, criterion = initialize_model(config['encoder_vocab_size'], config['decoder_vocab_size'], 0, 0, device, config)
    train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, 5, 1, device, fr_sp, 0)
