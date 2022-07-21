# Conditional Generative Adversarial Networks

### Demo notebook with code to generate Fashion MNIST images conditioned on class labels : [https://colab.research.google.com/drive/1o4eYZaDKc8Y3Nj4xmIhCyZ5g8ejDeERe?usp=sharing](https://colab.research.google.com/drive/1o4eYZaDKc8Y3Nj4xmIhCyZ5g8ejDeERe?usp=sharing)

### Genearator model

The generator model takes as input a point in the latent space and outputs a single 28×28 grayscale image. 

This is achieved by using a fully connected layer to interpret the point in the latent space and provide sufficient activations that can be reshaped into many copies (in this case 128) of a low-resolution version of the output image (e.g. 7×7). 

This is then upsampled twice, doubling the size and quadrupling the area of the activations each time using **transpose convolutional layers.** 

The model uses best practices such as the LeakyReLU activation, a kernel size that is a factor of the stride size, and a hyperbolic tangent (tanh) activation function in the output layer.

In this example the GAN’s are conditioned based off the class labels. This is done by embedding the class labels(0 - 9) into a Vector space 

<aside>
💡 The class label is then passed through an Embedding layer with the size of 50. This means that each of the 10 classes for the Fashion MNIST dataset (0 through 9) will map to a different 50-element vector representation that will be learned by the discriminator model.

</aside>

<aside>
💡 The output of the embedding is then passed to a fully connected layer with a linear activation. Importantly, the fully connected layer has enough activations that can be reshaped into one channel of a 28×28 image. The activations are reshaped into single 28×28 activation map and concatenated with the input image. This has the effect of looking like a two-channel input image to the next convolutional layer.

</aside>

---

Generative adversarial nets can be extended to a conditional model if both the generator and discriminator
are conditioned on some extra information y. y could be any kind of auxiliary information,
such as class labels or data from other modalities.

We can perform the conditioning by feeding y
into the both the discriminator and generator as additional input layer.

In the generator the prior input noise pz(z), and y are combined in joint hidden representation, and
the adversarial training framework allows for considerable flexibility in how this hidden representation
is composed. 1

In the discriminator x and y are presented as inputs and to a discriminative function (embodied
again by a MLP in this case).(Multi layer perceptron)

![Untitled](https://user-images.githubusercontent.com/72121513/180180343-27ffa1fd-029a-4451-9189-05b3f92bed33.png)


## Experimental Results

We trained a conditional adversarial net on MNIST images conditioned on their class labels, encoded as one-hot vectors

In the generator net, a noise prior z with dimensionality 100 was drawn from a uniform distribution
within the unit hypercube. Both z and y are mapped to hidden layers with Rectified Linear Unit
(ReLu) activation [4, 11], with layer sizes 200 and 1000 respectively, before both being mapped to
second, combined hidden ReLu layer of dimensionality 1200. We then have a final sigmoid unit
layer as our output for generating the 784-dimensional MNIST samples.

The discriminator maps x to a maxout [6] layer with 240 units and 5 pieces, and y to a maxout layer
with 50 units and 5 pieces. Both of the hidden layers mapped to a joint maxout layer with 240 units
and 4 pieces before being fed to the sigmoid layer.

The precise architecture of the discriminator
is not critical as long as it has sufficient power; we have found that maxout units are typically well
suited to the task.)

*The conditional adversarial net results that we present are comparable with some other network
based, but are outperformed by several other approaches – including non-conditional adversarial
nets. We present these results more as a proof-of-concept than as demonstration of efficacy, and
believe that with further exploration of hyper-parameter space and architecture that the conditional
model should match or exceed the non-conditional results.*

> GAN’s might become unstable during training. So certain architectures are used to imporve.
> 

 

## To explore

Diffusion models are genearative models that work on a different prinicple from GAN’s

## Multi-modal Learning

For instance in the case of
image labeling there may be many different tags that could appropriately applied to a given image,
and different (human) annotators may use different (but typically synonymous or related) terms to
describe the same image.

One way to help address the first issue is to leverage additional information from other modalities:
for instance, by using natural language corpora to learn a vector representation for labels in which
geometric relations are semantically meaningful. When making predictions in such spaces, we benefit
from the fact that when prediction errors we are still often ‘close’ to the truth (e.g. predicting
’table’ instead of ’chair’), and also from the fact that we can naturally make predictive generalizations
to labels that were not seen during training time.

---

<aside>
💡 There are many ways to encode and incorporate the class labels into the discriminator and generator models. A best practice involves using an embedding layer followed by a fully connected layer with a linear activation that scales the embedding to the size of the image before concatenating it in the model as an additional channel or feature map.

</aside>

## GAN Hacks

> https://github.com/soumith/ganhacks
> 

---

## Maxout

[https://medium.com/@rahuljain13101999/maxout-learning-activation-function-279e274bbf8e](https://medium.com/@rahuljain13101999/maxout-learning-activation-function-279e274bbf8e)

A type of Node in Neural Nework

A maxout layer is simply a layer where the activation function is the max of the inputs. As stated in the paper, even an MLP with 2 maxout units can approximate any function. They give a couple of reasons as to why maxout may be performing well, but the main reason they give is the following --

Dropout can be thought of as a form of model averaging in which a random subnetwork is trained at every iteration and in the end the weights of the different random networks are averaged. Since one cannot average the weights explicitly, an approximation is used. This approximation is *exact* for a linear networkIn maxout, they do not drop the inputs to the maxout layer. Thus the identity of the input outputting the max value for a data point remains unchanged. Thus the dropout only happens in the linear part of the MLP but one can still approximate any function because of the maxout layer.As the dropout happens in the linear part only, they conjecture that this leads to more efficient model averaging as the averaging approximation is exact for linear networks.

![Untitled_2](https://user-images.githubusercontent.com/72121513/180180443-93879266-e0b8-4ca5-8829-20ced899c7e7.png)


Thus, it is found that a **Maxout layer consisting of two Maxout units can approximate any continuous function *f*(*v*) arbitrarily well.**

![Untitled_3](https://user-images.githubusercontent.com/72121513/180180497-6902ee91-7b81-490a-8146-486cc45ed4f2.png)
