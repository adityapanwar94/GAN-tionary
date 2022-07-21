# Pix2Pix GAN - Image-to-Image Translation with Conditional Adversarial Networks

## Introduction

Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input image into a corresponding output image.

Image-to-image translation is the task of translating one possible representation of a scene into another.

Traditionally, each of these tasks has been tackled with separate, special-purpose machinery despite the fact that the setting is always the same: predict pixels from pixels.

A naive approach of using an CNN to minimize the Euclidean distance between predicted images and ground truth tends to produce blurry results. 

GAN’s can tackle this problem by optimizing for a common goal of making the output images indistinguishable from reality.

Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.

<aside>
💡 *Pix2PixGAN proposes a general model framework for any image-image translation task. So this can be utilised for any appication that we need to work on.*

</aside>

## Method

Conditional GANs learn a mapping from observed image x and random noise vector z, to y {x,z} - y
The generator G is trained to produce outputs that cannot be distinguished from “real” images by an adversarially trained
discriminator, D, which is trained to do as well as possible at detecting the generator’s “fakes”.

Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as **L2 distance**

The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but
also to be near the ground truth output in an L2 sense. The paper also explores this option, using L1 distance rather than L2 as
L1 encourages less blurring.

---

## Architecture

## Generator

Previous approaches have used an encoder-decoder network. In such a network, the input is passed through a series of layers
that progressively downsample, until a bottleneck layer, at which point the process is reversed. Such a network requires
that all information flow pass through all the layers, including the bottleneck.

For many image translation problems, there is a great deal of **low-level information shared
between the input and output**, and it would be desirable to **shuttle this information directly** across the network.

To give the generator a means to circumvent the bottleneck for sharing low-level information the model adds **skip-connections**, 

similar to an **U-Net**.

Each skip connection simply concatenates all channels at layer i with those at layer n - i.

![Untitled](https://user-images.githubusercontent.com/72121513/180185142-48ce6e16-7837-424f-8d59-bf0be3d25168.png)

---

## Discriminator

It is known that L1 and L2 loss tend to produce blurry results. These losses fail to encourage highfrequency crispness. 

In many cases however, they accurately capture the low frequencies.

This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to
force low-frequency correctness.

This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness. 

Therefore, this paper proposes **PatchGAN**, This discriminator tries to classify if each **N x N** patch in an image is real or fake. The model runs this **discriminator convolutionally across the image**, averaging all responses to provide the ultimate output of D.

The paper demonstrates that N can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images

---

## Optimization

Follows the standard approach, alternates between one gradient descent step on D, then one step on G.

As suggested in the original GAN paper, rather than training G to minimize log(1 - D(x;G(x; z)), we instead train to maximize logD(x;G(x; z))

The objective is divided by 2 while optimizing D to slow down the rate at which D learns relative to G.

Uses minibatch SGD and apply the Adam method, with a learning rate of 0:0002, and momentum.

The paper applies batch normalization  using the statistics of the test batch, rather than aggregated statistics of the training batch.

The paper claims that decent results can often be obtained even on small datasets (hundreds of images).

---

### Results

When both U-Net and encoder-decoder are trained with an L1 loss, the U-Net again achieves the superior results.

The results in this paper suggest that conditional adversarial networks are a promising approach for many image-to-image translation tasks.

---

## Refernces

[Image-to-Image Translation with Conditional Adversarial Networks](https://phillipi.github.io/pix2pix/)

Tensorflow implementation

[pix2pix: Image-to-image translation with a conditional GAN | TensorFlow Core](https://www.tensorflow.org/tutorials/generative/pix2pix)

Pytorch Implementation

[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

---

[How to Develop a Pix2Pix GAN for Image-to-Image Translation - Machine Learning Mastery](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/)

[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
