import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math


def dot_product(W_i, s_i):
    return jnp.dot(W_i, s_i)

batched_dot_product = jax.vmap(dot_product)

def hermitian_product(x,y):
    return jnp.dot(jnp.conj(x), y)

batched_hermitian_product = jax.vmap(hermitian_product)


@vmap
def batch_block_diag(W):
    return jax.scipy.linalg.block_diag(*W)

def train_and_plot(mu):
    noise = 0.
    dim = 1 #Dimensionality of s_t (1D)
    T = 30
    n = T
    b_ = int(1e3) #Batch size

    def batched_get_seq(b, n=n, dim=dim):
        key = jax.random.PRNGKey(4)

        s_0 = jnp.ones(dim) # / jnp.sqrt(dim)

        s_0 = jnp.tile(s_0, (b, 1))
        
        sequence = [s_0]

        # Generate random angle between 0 and 2*pi 
        #theta = np.random.uniform(0, 2*np.pi, (b,dim))
        
        theta = np.random.uniform(0, 2*np.pi, (b,dim))
        
        # Create complex number with modulus 1 and random phase
        W = jnp.exp(1j * theta / mu)  
        s = s_0 
        for t in range(n):
            eps = jnp.array(np.random.randn(b, dim))
            s = W * s + noise * eps
            sequence.append(s)
        return jnp.array(sequence).transpose((1, 0, 2)), W

    def tensordot(X_T, X):
        return jnp.tensordot(X_T,X, axes=((-1), (0)))


    mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_), k=0)

    # Expand the mask to match the batch size
    mask = jnp.expand_dims(mask, axis=0)  # shape: (1, d, d)
    mask = jnp.tile(mask, (b_, 1, 1))  # shape: (b, d, d)
    up_diag = jnp.eye(T, k=1)

    D, W = batched_get_seq(b_, n=n, dim=dim)
    seq = D[:,:-1,:]

    @jax.jit
    def linear_attention(params, s, n_layers=1, layer=1):
        p = params
        attention_scores = jnp.matmul(jnp.conj(s), s.transpose((0, 2, 1)))
        attention_scores = attention_scores * up_diag * p + attention_scores * p * mask
        attended_values = jnp.matmul(attention_scores, s)
        return attended_values
    



    @jax.jit
    def stack_linear_attention(params_list, s):
        output = jnp.zeros(s.shape)
        p = params_list[-1]
        params = p
        output = output + linear_attention(params, s, n_layers=1, layer=1)
        return output 

    @jax.jit
    def loss(params_list, seq, D):
        predictions = stack_linear_attention(params_list, seq)#[:,-1,:]
        y = predictions[:,:-1] - D[:,2:,:]
        return (jnp.mean(jnp.abs(y) ** 2)* dim)

    def return_param_list(num_heads=1, dimension=dim, scale=1e-2, use_full_params=True): 
        params_list = []
        p_init = jnp.zeros((T,T))
        params_list += [p_init]
        return params_list

    params_list = return_param_list(num_heads=dim, use_full_params=False)

    grad_loss = jax.jit(jax.grad(loss))

    # Hyperparameters
    learning_rate = 5e-1

    # Initialize the optimizer
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)

    @jit
    def update(params, opt_state, x, y):
        grads = grad_loss(params, x, y)
        return opt_update(0, grads, opt_state)

    # Training loop
    opt_state = opt_init(params_list)

    losses = []

    seq = D[:,:-1,:]
    targ = D[:,-1,:]

    num_epochs = 100000

    for epoch in range(num_epochs):
        params = get_params(opt_state)
        l = loss(params, seq, D)
        if epoch % 100 == 0:
            print('loss: ', l)
        if len(losses) > 0 and losses[-1] < 1e-3:
           break
        losses.append(l)
        opt_state = update(params, opt_state, seq, D)

    final_params = get_params(opt_state)
    mask_p = jnp.expand_dims(jnp.tril(jnp.ones((T, T), dtype=jnp.bool_), k=0), axis=0)
    p_opt = final_params[-1] * mask_p[0]

    np.save('results/noisy_pe/pe_eps_mu_%s.npy' %(mu), p_opt)
    plt.figure(figsize=(5, 5))
    plt.imshow((p_opt)[:-1,:-1])
    plt.axis('off')
    plt.savefig('figures/pe_eps_mu_%s.pdf' %(mu))

    plt.figure(figsize=(3, 3))
    plt.plot((p_opt)[:-1,:-1][-1])
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/proj_pe_eps_mu_%s.pdf' %(mu))


for mu in [50, 100, 200, 300]:
    print('mu = ', mu)
    train_and_plot(mu)
 