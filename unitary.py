import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 

# Given parameters
d = 10
T = 20
learning_rate = 0.01
num_iterations = 100
alpha = .1

# Initialize A, B, P randomly
key = jax.random.PRNGKey(0)  # PRNG key for reproducibility
key, *subkeys = jax.random.split(key, 4)
A = alpha * jax.random.normal(subkeys[0], (d, d)) 
B = alpha * jax.random.normal(subkeys[1], (d, d))
P = alpha * jax.random.normal(subkeys[2], (T, T))

# Loss function
def loss_function(C, p):
    d = C.shape[0]
    p_norm_squared = jnp.sum(p ** 2)
    C_norm_squared = jnp.sum(C ** 2)
    p_T_minus_1_squared = p[-2] ** 2 if p.shape[0] > 1 else 0
    sum_C_transpose_C = jnp.sum(C.T @ C)
    trace_C = jnp.trace(C)
    p_T = p[-1]
    loss = (p_norm_squared * C_norm_squared) + (p_T_minus_1_squared * sum_C_transpose_C) - (2 * trace_C * p_T) + d
    return loss

def new_loss_function(A, B, p):
    C = B.T @ A
    return loss_function(C, p)

def final_loss(A, B, P):
    total_loss = 0
    for t in range(T-1):
        p_t = P[t, :t+2]
        total_loss += new_loss_function(A, B, p_t)
    return total_loss

# Gradient function
grad_loss = jax.jit(jax.grad(final_loss, argnums=(0, 1, 2)))

# Gradient descent loop
for i in range(num_iterations):
    grad_A, grad_B, grad_P = grad_loss(A, B, P)
    A -= learning_rate * grad_A
    B -= learning_rate * grad_B
    P -= learning_rate * grad_P
    
    if i % 10 == 0:  # Print loss every 10 iterations
        print(f"Iteration {i}: Loss = {final_loss(A, B, P)}")

# Final parameters and loss
print("Final A:", A)
print("Final B:", B)
print("Final P:", P)
print("Final Loss:", final_loss(A, B, P))
np.save('tensors/A_unitary.npy', A)
np.save('tensors/B_unitary.npy', B)
np.save('tensors/P_unitary.npy', P)


C = np.dot(B.T, A)

p=1.1
# Set up the matplotlib figure and axes
fig, axs = plt.subplots(1, 4, figsize=(5*p, 2.5*p))

# Plot for Matrix A
axs[0].imshow(A, cmap='cividis', interpolation='nearest')
axs[0].set_title('A')
axs[0].set_xlabel('Dimension')
axs[0].set_ylabel('Heads')

# Plot for Matrix B
axs[1].imshow(B, cmap='cividis', interpolation='nearest')
axs[1].set_title('B')

# Plot for B^T * A
axs[2].imshow(C, cmap='cividis', interpolation='nearest')
axs[2].set_title('$B^{T} A$')

mask = np.triu(np.ones_like(P), k=2) # k=1 starts the mask above the diagonal

# Apply the mask to set these elements to zero
P_masked = np.where(mask, 0, P)
# Plot for Matrix P
axs[3].imshow(P_masked[:-1,:-1], cmap='cividis', interpolation='nearest')
axs[3].set_title('P')

# Show the plots
plt.savefig('figures/unitary.pdf')