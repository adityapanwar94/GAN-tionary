# Generative adversarial Networks

We simultaneously train two models:

A generative model G
that captures the data distribution, and a discriminative model D that estimates
the probability that a sample came from the training data rather than G.

The training procedure for G is to maximize the probability of D making a mistake.

- G should be able to produce data that tricks D such that D estimates that the data came from training data
- In the case where G and D are defined
by multilayer perceptrons, the entire system can be trained with backpropagation.

The special case when the generative model generates samples
by passing random noise through a multilayer perceptron, and the discriminative model is also a
multilayer perceptron. We refer to this special case as adversarial nets. In this case, we can train
both models using only the highly successful backpropagation and dropout algorithms [17] and
sample from the generative model using only forward propagation. No approximate inference or
Markov chains are necessary.

## ****The Generator Model****

The generator model takes a fixed-length random vector as input and generates a sample in the domain.

The vector is drawn from randomly from a Gaussian distribution, and the vector is used to seed the generative process. After training, points in this multidimensional vector space will correspond to points in the problem domain, forming a compressed representation of the data distribution.

This vector space is referred to as a latent space, or a vector space comprised of [latent variables](https://en.wikipedia.org/wiki/Latent_variable). Latent variables, or hidden variables, are those variables that are important for a domain but are not directly observable.

In the case of GANs, the generator model applies meaning to points in a chosen latent space, such that new points drawn from the latent space can be provided to the generator model as input and used to generate new and different output examples.

After training, the generator model is kept and used to generate new samples

![Untitled](https://user-images.githubusercontent.com/72121513/180178862-5e388994-1cee-43ee-90f3-9ede99e784ff.png)


## ****The Discriminator Model****

The discriminator model takes an example from the domain as input (real or generated) and predicts a binary class label of real or fake (generated).

The real example comes from the training dataset. The generated examples are output by the generator model.

The discriminator is a normal (and well understood) classification model.

After the training process, the discriminator model is discarded as we are interested in the generator.

---

Sometimes, the generator can be repurposed as it has learned to effectively extract features from examples in the problem domain. Some or all of the feature extraction layers can be used in transfer learning applications using the same or similar input data.


## ****GANs as a Two Player Game****

Generative modeling is an unsupervised learning problem, although a clever property of the GAN architecture is that the training of the generative model is framed as a supervised learning problem.

The two models, the generator and discriminator, are trained together. The generator generates a batch of samples, and these, along with real examples from the domain, are provided to the discriminator and classified as real or fake.

The discriminator is then updated to get better at discriminating real and fake samples in the next round, 

and importantly, the generator is updated based on how well, or not, the generated samples fooled the discriminator.


![Untitled_2](https://user-images.githubusercontent.com/72121513/180178424-d87117b9-36eb-47e8-8e9a-982a47d4deb5.png)

When the discriminator successfully identifies real and fake samples, it is rewarded or no change is needed to the model parameters, whereas the generator is penalized with large updates to model parameters.

Alternately, when the generator fools the discriminator, it is rewarded, or no change is needed to the model parameters, but the discriminator is penalized and its model parameters are updated. 

## Conditional GAN’s

An important extension to the GAN is in their use for conditionally generating an output.

The generative model can be trained to generate new examples from the input domain, where the input, the random vector from the latent space, is provided with (conditioned by) some additional input.

The additional input could be a class value, such as male or female in the generation of photographs of people, or a digit, in the case of generating images of handwritten digits.

**References**

https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/

https://arxiv.org/abs/1406.2661
