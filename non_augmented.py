import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import jax
import jax.numpy as jnp
from scipy.stats import ortho_group  # Requires version 0.18 of scipy
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, print_attention=False, init_same=False):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size 

        self.values = nn.Linear(embed_size, embed_size * heads)
        self.keys = nn.Linear(embed_size, embed_size * heads)
        self.queries = nn.Linear(embed_size, embed_size * heads)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.print_attention = print_attention
        self.init_same = init_same
        if self.init_same:
            for q in [self.values, self.keys, self.queries]:
                initial_weights = q.weight.data[:embed_size, :].clone() 
                initial_bias = q.bias.data[:embed_size].clone() 

                # Copy the initial segment across all heads
                for h in range(heads):
                    q.weight.data[h*embed_size:(h+1)*embed_size,:] = initial_weights + .05 * torch.randn(embed_size, embed_size)
                    q.bias.data[h*embed_size:(h+1)*embed_size] = initial_bias + .05 * torch.randn(embed_size)

    def forward(self, values, keys, query, pos, mask):
        # Get number of training examples

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention_tok = energy / (self.embed_size ** (1 / 2))

        N = query.shape[0]
        values_pos, keys_pos, query_pos = pos, pos, pos
        value_len, key_len, query_len = values_pos.shape[1], keys_pos.shape[1], query_pos.shape[1]

        values_pos = self.values(values_pos)  # (N, value_len, embed_size)
        keys_pos = self.keys(keys_pos)  # (N, key_len, embed_size)
        queries_pos = self.queries(query_pos)  # (N, query_len, embed_size)

        values_pos = values_pos.reshape(N, value_len, self.heads, self.head_dim)
        keys_pos = keys_pos.reshape(N, key_len, self.heads, self.head_dim)
        queries_pos = queries_pos.reshape(N, query_len, self.heads, self.head_dim)

        energy_pos = torch.einsum("nqhd,nkhd->nhqk", [queries_pos, keys_pos])

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy_pos = energy_pos.masked_fill(mask == 0, float("-1e20"))

        attention_pos = torch.softmax(energy_pos / (self.embed_size ** (1 / 2)), dim=3)

        attention = attention_tok * attention_pos

        # attention shape: (N, heads, query_len, key_len)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values ])

        out = out.sum(2)
        out = self.fc_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, print_attention=False):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, print_attention=print_attention)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, pos, mask):
        attention = self.attention(value, key, query, pos, mask)
        out= self.norm1(attention)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        print_attention=False
    ):

        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    print_attention=print_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        pos = self.position_embedding(positions)
        out = x
        for layer in self.layers:
            out = layer(out, out, out, pos, mask)
        return out




class Transformer(nn.Module):
    def __init__(
        self,
        dim_in=2,
        embed_size=16,
        num_layers=1,
        forward_expansion=8,
        heads=1,
        dropout=0,
        device=device,
        max_length=100,
        print_attention=False
    ):

        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.fc_out = nn.Linear(embed_size, dim_in)
        self.decoder = Decoder(
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            print_attention=print_attention
        )

        self.device = device


    def make_mask(self, trg):
        N, trg_len, _ = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src):
        _, _, d = src.shape
        n = self.embed_size // d
        src = src.repeat(1, 1, n)
        mask = self.make_mask(src)
        out = self.decoder(src, mask)
        return self.fc_out(out)

sigma = .0  

def dot_product(W_i, s_i):
    return jnp.dot(W_i, s_i)

batched_dot_product = jax.vmap(dot_product)

def batched_get_seq(b, n, dim):
    key = jax.random.PRNGKey(2)
    s_0 = jnp.ones(dim)
    s_0 /=((s_0 ** 2).sum() ** .5)
    s = jnp.tile(s_0, (b,1))

 
    sequence = [s]
    W = jnp.array(ortho_group.rvs(dim=dim, size=b)) if dim > 1 else jnp.sign(jnp.array(np.random.rand(b, dim, dim) - .5))
    
    for t in range(n):
        # Generate a random orthogonal matrix W
        # Update s_{t+1} = W s_t + epsilon_t
        s = batched_dot_product(W, s)
        sequence.append(s)
    return jnp.array(sequence).transpose((1, 0, 2)), W



class CustomDataset(Dataset):

    def __init__(self, numpy_array):
        self.data = torch.from_numpy(numpy_array).float()  # Convert to PyTorch tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index, :-1, :]  # Input features
        target = self.data[index, 1:, :]   # Target labels

        return sample, target


def train(dim_in=5, n=50, embed_extension=2, heads=64, lr=1e-2, num_epochs=200, batch_size=1024, save_folder='results'):
    """
    Train a transformer model on a dataset of sequences.
    
    dim_in: Dimensionality of the input sequences
    n: Length of the sequences
    embed_extension: Extension factor for the embedding size
    heads: Number of heads in the self-attention mechanism
    lr: Learning rate
    num_epochs: Number of epochs
    batch_size: Batch size
    save_folder: Folder to save the results
    
    Returns None.
    """

    save_output = 'dim_in_%s_n_%s_embed_extension_%s_heads_%s_lr_%s_num_epochs_%s_batch_size_%s' % (dim_in, n, embed_extension, heads, lr, num_epochs, batch_size)

    D_train, W_train = batched_get_seq(2 ** 14, n=n, dim=dim_in)
    D_test, W_test = batched_get_seq(2 ** 10, n=n, dim=dim_in)


    numpy_array = np.array(D_train)  

    custom_dataset = CustomDataset(numpy_array)

    batch_size = min(batch_size, D_train.shape[0])

    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


    numpy_array = np.array(D_test) 

    custom_dataset = CustomDataset(numpy_array)

    test_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)


    embed_size = embed_extension * dim_in
    num_layers = 1
   
    criterion = nn.MSELoss()

    model = Transformer(
            dim_in=dim_in,
            embed_size=embed_size,
            num_layers=num_layers,
            forward_expansion=1,
            heads=heads,
            print_attention=False).to(device)


    train_losses = []
    test_losses = []

    optimizer = optim.Adam(model.parameters(), lr=lr)



    def train(epoch):
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs[:,:], targets[:,:])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            train_losses.append(loss.item())
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')

    def test(epoch):
        if epoch % 10 == 0:
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs[:,:], targets[:,:])

            test_losses.append(loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {loss.item():.4f}')

    for epoch in range(num_epochs):
        train(epoch)
        test(epoch)

    np.save('%s/losses_%s.npy' % (save_folder, save_output), np.array([train_losses, test_losses]))
    torch.save(model.state_dict(), '%s/model_%s.pt' % (save_folder, save_output))
    print('done')

