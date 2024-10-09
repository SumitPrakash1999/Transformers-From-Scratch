import torch
import sacrebleu
from utils import create_data_loader, calculate_bleu, epoch_time
from transformer import Transformer
from encoder import Encoder
from decoder import Decoder

# Function to evaluate the model
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            trg_input = trg[:, :-1]
            trg_pad_mask = (trg_input != 0).unsqueeze(1).unsqueeze(2)
            
            output = model(src, src_mask, trg_input, trg_pad_mask)
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            
            predictions.append(output.argmax(dim=-1).cpu().numpy())
            targets.append(trg.cpu().numpy())
    
    return epoch_loss / len(iterator), predictions, targets

# Test the model and write BLEU scores for each sentence in the test set to a file
def test_model(model, test_loader, criterion, trg_vocab, trg_pad_idx, device, output_file="testbleu.txt"):
    test_loss, predictions, targets = evaluate(model, test_loader, criterion, device)
    
    # Open the output file to write the sentence-level BLEU scores
    with open(output_file, "w") as f:
        for i, (pred, target) in enumerate(zip(predictions, targets), start=1):
            # Calculate BLEU score for this sentence pair
            pred_sentence = ' '.join([trg_vocab.IdToPiece(int(idx)) for idx in pred if idx != trg_pad_idx])
            target_sentence = ' '.join([trg_vocab.IdToPiece(int(idx)) for idx in target if idx != trg_pad_idx])
            sentence_bleu = sacrebleu.corpus_bleu([pred_sentence], [[target_sentence]]).score
            # Write "Sentence1", "Sentence2", etc., and the BLEU score to the file
            f.write(f"Sentence{i} {sentence_bleu:.2f}\n")
    
    # Calculate and print overall test BLEU score
    test_bleu = sacrebleu.corpus_bleu(
        [' '.join([trg_vocab.IdToPiece(int(idx)) for idx in pred if idx != trg_pad_idx]) for pred in predictions],
        [[' '.join([trg_vocab.IdToPiece(int(idx)) for idx in target if idx != trg_pad_idx]) for target in targets]]
    ).score
    print(f"Test BLEU: {test_bleu:.2f}")

# Load and evaluate the model
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

    # Initialize the model
    model = Transformer(Encoder(config), Decoder(config)).to(device)
    model.load_state_dict(torch.load('transformer.pt'))
    model.eval()

    # Load test dataset
    test_loader = create_data_loader(test_src, test_trg, en_sp.encode_as_ids, fr_sp.encode_as_ids, 128)
    
    # Test and evaluate
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    test_model(model, test_loader, criterion, fr_sp, 0, device)
