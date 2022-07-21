# LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS (BIG-GAN)

## SCALING UP GANS

The paper demonstrates that GANs benefit dramatically from scaling, and trains models with **two to four times** as many parameters and **eight times** the batch size compared to prior art.

As a side effect of our modifications, our models become amenable to the “**truncation trick**,” a simple sampling technique that allows explicit, fine-grained control of the tradeoff
between sample variety and fidelity.

We discover instabilities specific to large scale GANs, and characterize them empirically. Leveraging insights from this analysis, we demonstrate that a combination of novel and
existing techniques can reduce these instabilities.

As a baseline, the paper employs the SA-GAN architecture which uses the **hinge loss**.

BigGAN applies a self-attention layer in the 64x64 resolution of both the generator and discriminator.

We provide class information to G with class-***conditional BatchNorm***  and to D with projection.

The latent vector Z and the class embedding y are passed through the generator with class-conditional BatchNorm in multiple scales and help enforce class consistency in the generated image.

The optimization settings follow Zhang et al. (2018) (notably employing Spectral Norm in G) with the modification that we ***halve*** the learning
rates and ***take two D steps per G step***.

The Optimization employs ***Spectral Norm*** in G

---

### Insigths

Increasing the batch size results in a increase in performance.

Simply increasing the batch size by a factor of 8 improves the state-of-the-art IS(Inception score) by **46%**.

One notable side effect of this scaling is that our models reach better final performance in fewer iterations, but become unstable and undergo complete training collapse.

Increasing the width (number of channels) in each layer by 50%, approximately doubling the number of parameters in both models leads to a further IS improvement of **21%**.

The paper adds direct skip connections (skip-z) from the noise vector z to multiple layers of G rather than just
the initial layer. The intuition behind this design is to allow G to use the latent space to directly influence
features at different resolutions and levels of hierarchy. Skip-z provides a modest performance
improvement of around **4%**, and improves training speed by a further **18%.**

For skip-z connection, the model splits the latent vector into chunks and each chunk is fed into the residual block concatenated with the class embedding. 

Class embeddings c used for the conditional BatchNorm layers in G contain a large number of weights. Instead of having a separate layer for each embedding a **shared embedding** is used. This reduces computation and memory costs, and improves training speed by **37%.**

The Models undergo training collapse, necessitating early stopping in practice. It is possible to enforce stability by strongly constraining D, but
doing so incurs a dramatic cost in performance.

![Untitled](https://user-images.githubusercontent.com/72121513/180184531-b3ef505f-d209-47ff-a141-e7a946c99d23.png)

---

Generator architecture, G residual block, D residual block architecture respectively.

*The latent vector z and class embedding are concatenated and fed into each residual block in the generator. Each residual block fundamentally consists of two convolutions and a skip connection. The class embedding is concatenated before the final layer in the discriminator.*

## Truncation trick

The Best results have come from using a different latent distribution for sampling than was used in training.

<aside>
💡 Taking a model trained with z  N(0; I) and sampling z from a truncated normal (where values which fall outside a range are resampled to fall inside that range) 
immediately provides a boost to IS and FID(performance metrics).

</aside>

## Glossary

### Hinge Loss

This hinge loss for adversarial training was used in the SA-GAN, and training the generator and discriminator is a little different. The discriminator is trained to predict D(x)>1 for real data and D(G(z))<-1 for fake data, while the generator only aims at pushing the score of the fake images to be D(G(z))>0.

### ****Conditional Batch Normalization****

In batch normalization, the parameters of the batch normalization layer are set as the mean and variance of the batch elements. In conditional batch normalization, the mean and variance are set to outputs of a neural network. In this case, it is conditioned based on the latent slice z and class embedding. The only way the model gets knowledge about the class embedding and noise is through conditional batch normalization.

## Truncation Trick

Truncating a z vector by resampling the values with magnitude above a chosen threshold leads to improvement in individual
sample quality at the cost of reduction in overall sample variety.

---

## Code

Pytorch:

[https://github.com/ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch)

Tensorflow:

[https://github.com/taki0112/BigGAN-Tensorflow](https://github.com/taki0112/BigGAN-Tensorflow)

---

## References

[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)

[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)

[Key Concepts of BigGAN: Training and assessing large-scale image generation](https://medium.com/analytics-vidhya/key-concepts-of-biggan-training-and-assessing-large-scale-image-generation-4c8303dcf73f)
