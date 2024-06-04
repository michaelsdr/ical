import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 

# Given parameters
d = 10
H = 8
T = 20


# Initialize A, B, P randomly
key = jax.random.PRNGKey(0)  # PRNG key for reproducibility
key, *subkeys = jax.random.split(key, 4)


def loss_function(C, p):
    d = C.shape[0]
    t1 = (C.T @ C).sum() # t = t' = T-1
    t2 = (C ** 2).sum() # t = t' != T-1
    t3 = (C.sum(-1)[::2] * jnp.diag(C, k =1)[::2] + C.sum(-1)[1::2] * jnp.diag(C, k =-1)[::2]).sum() # t = T-1, t'!= T-1 or t'= T-1, t != T-1
    t4 = jnp.diag(C**2, k =1)[::2].sum() + jnp.diag(C**2, k =-1)[::2].sum() # t != t', t!=T-1, t'!=T-1
    t5 = - 2 * jnp.trace(C) 
    p_no_2 = jnp.append(p[:-2], p[-1])
    return p[-2] ** 2 * t1 + ((p ** 2).sum() - p[-2] ** 2) * t2 + 2 * (t3 * ((p[-2] * p).sum() - p[-2] ** 2)) + ((p_no_2.sum() ** 2) - (p_no_2**2).sum()) * t4 + t5 * p[-1] + d


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

alpha = .1
A = alpha * jax.random.normal(subkeys[0], (H, d)) 
B = alpha * jax.random.normal(subkeys[1], (H, d))
P = alpha * jax.random.normal(subkeys[2], (T, T))

learning_rate = 1e-3
num_iterations = 100000

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
np.save('tensors/A_orthogonal.npy', A)
np.save('tensors/B_orthogonal.npy', B)
np.save('tensors/P_orthogonal.npy', P)


C = np.dot(B.T, A)

p=1.1
# Set up the matplotlib figure and axes
fig, axs = plt.subplots(1, 4, figsize=(5*p, 2.5*p))

# Plot for Matrix A
axs[0].imshow(A, cmap='cividis', interpolation='nearest')
axs[0].set_title('A')



# Plot for Matrix B
axs[1].imshow(B, cmap='cividis', interpolation='nearest')
axs[1].set_title('B')

# Plot for B^T * A
axs[2].imshow(C, cmap='cividis', interpolation='nearest')
axs[2].set_title('$B^{T} A$')

mask = np.triu(np.ones_like(P), k=2)  # k=1 starts the mask above the diagonal

# Apply the mask to set these elements to zero
P_masked = np.where(mask, 0, P)
# Plot for Matrix P
axs[3].imshow(P_masked[:-1,:-1], cmap='cividis', interpolation='nearest')
axs[3].set_title('P')



# Show the plots
plt.savefig('figures/orthogonal.pdf')