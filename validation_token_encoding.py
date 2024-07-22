import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# Number of sequences to consider
n = 10000

# Sequence length
seq_lenght = 5

def get_batch(n=n, seq_lenght=seq_lenght):
    """Get a batch of sequences and a shuffled version for the experiment."""
    # Download necessary NLTK datasets
    nltk.download('gutenberg')
    nltk.download('punkt')

    # Load text

    text = gutenberg.raw('melville-moby_dick.txt') 
    sentences = sent_tokenize(text)
    random_sentences = sentences

    # Simplify each sentence 
    simple_sentences = ['.'.join(sent.split('.')[:2]) for sent in random_sentences]

    model_name = 'gpt2-large'  
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize and encode sentences in a batch

    input_ids = tokenizer(simple_sentences, return_tensors='pt', padding=True, truncation=True)

    input_ids_flatten = input_ids['input_ids'].flatten()
    input_ids_flatten = input_ids_flatten[torch.where(input_ids_flatten != input_ids_flatten.max())]

    r = input_ids_flatten.shape[0] % seq_lenght

    input_ids_flatten = input_ids_flatten[:-r]

    input_ids_to_embed = input_ids_flatten.reshape(-1, seq_lenght)

        
    batch_seq = model.transformer.wte(input_ids_to_embed[:,:]).float().detach()[:n]
    batch_seq = batch_seq / batch_seq.norm(dim=-1,keepdim=True) # Normalizing each token to unit norm

    N, T, d = batch_seq.shape

    tensor_reshaped = batch_seq.reshape(-1, d)

    # Permute rows
    permuted_indices = torch.randperm(N*T)
    tensor_permuted = tensor_reshaped[permuted_indices]

    # Reshape back to (N, T, d)
    rand_batch_seq = tensor_permuted.view(N, T, d)

    return batch_seq, rand_batch_seq


def fit_ar(sequence):
    """
    Fit an autoregressive model to a sequence.

    sequence: Sequence of shape (T, d)

    Returns the estimated A matrix that fits an AR process on sequence.
    """
    inputs = sequence[:-1]  # All elements except the last
    outputs = sequence[1:]  # All elements except the first

    A_estimated = torch.linalg.lstsq(inputs, outputs)[0].T
    
    return A_estimated

def generate(A_estimated, s0, n):
    """
    Generate a sequence from an autoregressive model.
    
    A_estimated: Estimated A matrix
    s0: Initial state
    n: Length of the sequence to generate
    
    Returns the generated sequence."""
    # Solve for A
    # Multiply with outputs (also transposed) and transpose the result for correct shape

    # Initialize the approximated sequence with the first element of the original sequence
    approx_sequence = [s0]

    # Compute the approximated sequence using the estimated A
    for _ in range(1, n):
        next_element = torch.matmul(A_estimated, approx_sequence[-1].unsqueeze(-1)).squeeze(-1)
        approx_sequence.append(next_element)

    # Convert the list of tensors to a tensor
    approx_sequence = torch.stack(approx_sequence)
    return approx_sequence


def plot_hist(batch_seq, rand_batch_seq):
    """
    Plot a histogram of the mean squared error of the original and shuffled sequences.
    
    batch_seq: Original batch of sequences
    rand_batch_seq: Shuffled batch of sequences
    
    Returns None."""
    approx_sequences = []

    for i in range(len(batch_seq)):
        sequence = batch_seq[i]
        A = fit_ar(sequence)
        approx_sequence = generate(A, sequence[0], batch_seq.shape[1])
        approx_sequences.append(approx_sequence)

    approx_sequences = torch.stack(approx_sequences)

    out = (approx_sequences - batch_seq) ** 2

    rand_approx_sequences = []
    for i in range(len(rand_batch_seq)):
        sequence = rand_batch_seq[i]
        A = fit_ar(sequence)
        approx_sequence = generate(A, sequence[0], rand_batch_seq.shape[1])
        rand_approx_sequences.append(approx_sequence)

    rand_approx_sequences = torch.stack(rand_approx_sequences)

    rand_out = (rand_approx_sequences - rand_batch_seq) ** 2

    out_mean = out.mean((1,2))
    rand_out_mean = rand_out.mean((1,2))

    plt.figure(figsize=(2.5, 2.5))
    plt.hist(out_mean[torch.where(out_mean > 1e-12)], bins=100, color='#007acc', alpha=0.7, label='Original')
    plt.hist(rand_out_mean[torch.where(rand_out_mean > 1e-12)], bins=100, color='#cc7000', alpha=0.5, label='Shuffled')
    plt.xlabel('MSE')  
    plt.ylabel('Frequency')  
    plt.legend() 
    plt.tight_layout()  
    plt.grid(axis='y', alpha=0.75)  
    plt.savefig('figures/hist.pdf')

if __name__ == '__main__':
    batch_seq, rand_batch_seq = get_batch(n, seq_lenght)
    plot_hist(batch_seq, rand_batch_seq)
