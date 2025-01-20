# Variational Autoencoder (VAE)

A **Variational Autoencoder (VAE)** is a generative model that learns to represent data in a latent space, enabling the generation of new data samples similar to the original dataset. It combines principles from autoencoders and variational inference.

## Key Components of VAE

1. **Encoder:** Maps input data \( x \) to a latent representation \( z \). Instead of mapping to a single point, the encoder predicts parameters of a probability distribution (typically Gaussian), allowing for variability in the latent space. This is represented as:

   \[
   q_\phi(z|x) = \mathcal{N}(z; \mu, \sigma^2)
   \]

   where \( \mu \) and \( \sigma^2 \) are the mean and variance predicted by the encoder network with parameters \( \phi \).

2. **Latent Space:** A continuous, multidimensional space where each point represents a possible configuration of the data. Sampling from this space allows the VAE to generate new data instances.

3. **Decoder:** Reconstructs data \( x' \) from the latent variable \( z \). The decoder defines the likelihood of the data given the latent variables:

   \[
   p_\theta(x|z)
   \]

   where \( \theta \) represents the decoder's parameters.

## Objective Function

The VAE is trained to maximize the Evidence Lower Bound (ELBO), which balances two objectives:

1. **Reconstruction Loss:** Ensures that the decoded output \( x' \) is similar to the input \( x \). This is typically measured using the expected log-likelihood:

   \[
   \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]
   \]

2. **Kullback-Leibler (KL) Divergence:** Regularizes the encoder by ensuring that the learned distribution \( q_\phi(z|x) \) is close to the prior distribution \( p(z) \) (often a standard normal distribution). This is given by:

   \[
   D_{KL}(q_\phi(z|x) \parallel p(z))
   \]

Combining these terms, the ELBO is:

\[
\mathcal{L}(\phi, \theta; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))
\]

Maximizing this objective ensures that the VAE learns a meaningful latent representation capable of generating data similar to the original input.

## Reparameterization Trick

To backpropagate through the sampling process, VAEs use the reparameterization trick. Instead of directly sampling \( z \) from \( q_\phi(z|x) \), we express \( z \) as:

\[
z = \mu + \sigma \odot \epsilon
\]

where \( \epsilon \sim \mathcal{N}(0, I) \) is sampled from a standard normal distribution, and \( \odot \) denotes element-wise multiplication. This allows gradients to flow through the network during training.

By learning to encode data into a probabilistic latent space and decode from it, VAEs can generate new, coherent data samples and are widely used in tasks like image generation and anomaly detection.


