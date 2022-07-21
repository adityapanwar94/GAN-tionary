## UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

- The **Discriminator** is a classifier that determines whether the given image is “real” and “fake”.
- The **Generator** takes a randomly generated noise vector as input data and feedback from the **Discriminator** and generates new images that are as close to real images as possible.
- The **Discriminator** uses the output of the **Generator** as training data.
- The **Generator** gets feedback from the **Discriminator**.
- These two models “battle” to each other. Each models becomes stronger in the process.
- The **Generator** keeps creating new images and refining its process until the **Discriminator** can no longer tell the difference between the generated images and the real training images.

![Untitled](https://user-images.githubusercontent.com/72121513/180181835-e30d8a35-ec49-44bc-87b8-c0973af5da34.png)


DCGAN implemented in the papers has *four* convolutional layers for the **Discriminator** and *four “*four fractionally-strided convolutions” layers for the **Generator.**

---

### ****The Discriminator Network****

The **Discriminator** is a 4 layers strided convolutions with batch normalization (except its input layer) and leaky ReLU activations.

**Discriminator** needs to output probabilities. For that, we use the *Logistic Sigmoid* activation function on the final logits.

### The Generator Network

The **Generator** is a 4 layers fractional-strided convolutions with batch normalization (except its input layer) and use **Hyperbolic Tangent (*tanh*)**
 activation in the final output layer and **Leaky ReLU** in rest of the layers.

![Untitled_2](https://user-images.githubusercontent.com/72121513/180181897-7d5a0988-23e2-4771-a74a-d41e21d25b4b.png)


**Training of DCGANs**

**The following steps are repeated in training**

- First the **Generator** creates some new examples.
- The **Discriminator** is trained using real data and generated data.
- After the **Discriminator** has been trained, both models are trained together.
- The **Discriminator**’s weights are frozen, but its gradients are used in the **Generator** model so that the **Generator** can update it’s weights.

---

# Thesis

- Replace any pooling layer with strided convolutions for discriminator, and use fractional-strided convolutions for generator. This way the network can learn its own spatial downsampling… why?
- eliminating all fully connected layers on top of convolutional features. Even for the last layer, the convolution layer is flattened and fed into a single sigmoid output.
- batch normalization is applied to applied to every layer except for generator output and discriminator input layer.
- ReLU is used in the generator except for the output layer which uses Tanh, because bounded activation allowed the model to learn more quickly to saturate.
- use LeakyReLU activation in the discriminator for all layers.

---

## APPROACH AND MODEL ARCHITECTURE

*As presented in the paper*

Core to our approach is adopting and modifying three recently demonstrated changes to CNN architectures.
The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic spatial
pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn
its own spatial downsampling. We use this approach in our generator, allowing it to learn its own
spatial upsampling, and discriminator.

Second is the trend towards eliminating fully connected layers on top of convolutional features.
The strongest example of this is global average pooling which has been utilized in state of the
art image classification models (Mordvintsev et al.). 

We found global average pooling increased model stability but hurt convergence speed. A middle ground of directly connecting the highest convolutional features to the input and output respectively of the generator and discriminator worked well. 

The first layer of the GAN, which takes a uniform noise distribution Z as input, could be called
fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional
tensor and used as the start of the convolution stack. For the discriminator, the last convolution layer
is flattened and then fed into a single sigmoid output. See Fig. 1 for a visualization of an example
model architecture.

Third is Batch Normalization (Ioffe & Szegedy, 2015) which stabilizes learning by normalizing the
input to each unit to have zero mean and unit variance. 

This proved critical to get deep generators to begin learning, preventing the generator from 

collapsing all samples to a single point which is a common failure mode observed in GANs. 

Directly applying batchnorm to all layershowever, resulted in sample oscillation and model instability. 

This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.

The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output
layer which uses the Tanh function. We observed that using a bounded activation allowed the model
to learn more quickly to saturate and cover the color space of the training distribution. Within the
discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work
well, especially for higher resolution modeling. This is in contrast to the original GAN paper, which
used the maxout activation

---

# **1. A Set of Constraints for Stable Training**

## **1.1. All convolutional net replaces deterministic spatial pooling functions (such as max pooling) with strided convolutions.**

- The generator learns its own spatial downsampling itself using convolution.
- Similarly, the discriminator learns its own spatial upsampling.

## **1.2. Eliminating Fully Connected Layers**

- The first layer of the generator, which takes a uniform noise distribution *Z* as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack.
- For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.
- But there are no fully connected layers for all hidden layers.

## **1.3. Batch Normalization (BN)**

- BN stabilizes learning by normalizing the input to each unit to have zero mean and unit variance.
- However, directly applying BN to all layers resulted in sample oscillation and model instability.
- This was avoided by not applying BN to the generator output layer and the discriminator input layer.

## **1.4. Activation Functions**

- The ReLU activation is used in the generator with the exception of the output layer which uses the Tanh function.
- Within the discriminator, it is found that the leaky rectified activation (LeakyReLU) works well.

---

<aside>
💡 There are still some forms of model instability remaining - It is  noticed as
models are trained longer they sometimes collapse a subset of filters to a single oscillating mode. Further work is needed to tackle this from of instability.

</aside>

---

## Resources / References

[gans-in-action/Chapter_4_DCGAN.ipynb at master · GANs-in-Action/gans-in-action](https://github.com/GANs-in-Action/gans-in-action/blob/master/chapter-4/Chapter_4_DCGAN.ipynb)

[Deep Convolutional Generative Adversarial Network | TensorFlow Core](https://www.tensorflow.org/tutorials/generative/dcgan)

[DCGAN Tutorial - PyTorch Tutorials 1.11.0+cu102 documentation](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

[Generative Adversarial Networks - The Story So Far](https://blog.floydhub.com/gans-story-so-far/#dcgan)

[Review: DCGAN - Deep Convolutional Generative Adversarial Network (GAN)](https://sh-tsang.medium.com/review-dcgan-deep-convolutional-generative-adversarial-network-gan-ec390cded63c)

---

### Concepts/ Glossary

### RELU vs LeakyRELU

If we use the ReLU activation function, sometimes the network gets stuck in a popular state called the **dying state**, and that’s because the network produces nothing but zeros for all the outputs.

### LeakyRELU

The output of the **Leaky ReLU** activation function will be positive if the input is positive, and it will be a controlled negative value if the input is negative. Negative value is control by a parameter called **alpha.**

![Untitled_3](https://user-images.githubusercontent.com/72121513/180181979-e8da066f-0bdd-4222-8ebe-6c243f48362c.png)


---

### ****Adaptive Moment Estimation(Adam) optimizer****

• Adam is different to classical stochastic gradient descent.

• Adam adds **Root Mean Square Deviation** (**RMSprop**) to the process by applying per parameter learning weights. It analyzes how fast the means of the weights are changing and adapts the learning weights.
