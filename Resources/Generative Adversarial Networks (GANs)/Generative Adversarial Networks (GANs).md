<img src = "Assets/Cover.jpg">

<br>

# Generative Adversarial Networks (GANs)

<br>
<p align = "justify">
Generative Adversarial Networks or GANs are bassed on a scenario in which the generator network competes against an adversary where, the generator directly produces samples, and its adversary, the discriminator attempts to distinguish between the samples drawn from the training data and the samples produced by the generator.</p>
<p align = "justify">
In other words, the generator takes in a noisy input and tries to produce realistic samples from it. Although it fails at first, it slowly picks up patterns and produce data samples, which are realistic enough to fool and bypass the discriminator. So these models, work and compete against each other till a certain level of perfection is reached.</p>

<br>
<div align = "center"> <img src = "Assets/GANs Architecture.png" width = 350> </div>
<br>

**Generative Modeling:**
<br>
<p align = "justify">
Generative modeling is an unsupervised learning task in machine learning that involves automatically discovering, learning the regularities or patterns in input data and develop some insights in such a way that the model can be used to generate or output new examples that plausibly could have been drawn from the original dataset.</p>
<p align = "justify">
GANs are a clever way of training a generative model by framing the problem as a supervised learning problem with two sub-models: Generator Model that generates new examples and the Discriminator Model that tries to classify examples as either real (from the domain) or fake (generated). The two models are trained together in a zero-sum game, adversarial, until the discriminator model is fooled about half the time, meaning the generator model is generating plausible examples.</p>

### Generative and Discriminative Models

<br>
<div align = "center"> <img src = "Assets/Generator Discriminator.png" width = 350> </div>
<br>
<br>

**Generative Models**

<br>
<p align = "justify">
The generated model takes a fixed-length random noisy vector as input and generates a sample from it. This vector is drawn randomly form a Guassian distribution and is used to seed the generation process. After training, points in this multidimensional vector space will correspond to points in the problem domain, forming a compressed representation of the data distribution.In the case of GANs, the generator model applies meaning to points in a chosen latent space, such that new points drawn from the latent space can be provided to the generator model as input and used to generate new and different output examples.<p>

> This vector space is referred to as a latent space, or a vector space comprised of [latent variables](https://en.wikipedia.org/wiki/Latent_variable). Latent variables, or hidden variables, are those variables that are important for a domain but are not directly observable.

<br>
<p align = "justify">
In simple terms a generative model is used to generate new plausible examples from the problem domain.</p>
<p align = "justify">
The role of the generator is generate fake examples that are so realistic that it becomes impossible for its adversary, the discriminator to distinguish between them and the real samples. Initially, the generator fails pretty badly at its job due to introduction of random noise and also because at the beginning the generator model is just learning what it has to generate. But over the time, with updation of its parameters, it slowly picks up patterns and bypasses the detection of the discriminator.</p>
<p align = "justify">
Suppose, the generator generates sample with a set of features that are fed to the discriminator, which classifies and detects whether they are real. The generator would want the output of this discriminator to be as close to 1, because that is the actual output by the discriminator in case of real samples.</p>
<p align = "justify">
From this, the cost function can be evaluated by taking 1 for real samples and 0 for the fake ones. This function is the used to update the parameters of the model, and improve it with each time step, to ultimately generate better samples.</p>
<br>

**Discriminative Models**

<br>
<p align = "justify">
Discriminative models, on the other hand, is used to classify examples  as real (from the domain) or fake (generated).</p>
<p align = "justify">
The discriminator basically acts as a classifier which classifies and detects real images from fake ones. This discriminator ouputs a value between 0 and 1, which reflects its confidence in the sample. The closer the value is to 1, the higher the probability the sample being real or drawn from the dataset.</p>
<p align = "justify">
In this case, we can calculate the cost quite simply by considering the predicted outputs of the discriminator and the actual class labels. From this cost function, we can similarly update the parameters of the discriminator, ultimately making it more strict with its judgement.</p>
<br>

> **Generator:** Takes random noise **z** as input and outputs an image **x**. Its parameters are tuned to get a high score from the discriminator on fake images that it generates.

> **Discriminator:**  Takes an image **x** as input and outputs a score which reflects its confidence that it is a real image. Its parameters are tuned to have a high score when it is fed by a real image, and a low score when a fake image is fed from the generator.

<br>

### Training Procedure

<br>
<p align = "justify">
Binary Cross Entropy (BCE) is extremely useful for training GANs. The main purpose of this function is the utility it has for classification tasks for the prediction of real or fake data. </p>
<br>

**Cost Function**

$$
J(\theta) = -1/m * \Sigma[y(i)logh(x(i),\theta) + (1-y(i))log(1-h(x(i),\theta))]
$$

where,

- **-1/m∗Σ** represents the average loss of the whole batch. The negative sign at the beginning of the equation is to symbolize and always ensure that the cost computed is greater than or equal to zero. Our main objective is to reduce the cost function for producing better results.
- **h** is the representation of the predictions made.
- **y(i)** represents the labels of the computations. **y(0)** could stand for fake images, while **y(1)** could represent real images.
- **x(i)** are the features that are calculated.
- $\theta$ is a representation of the parameters that need to be calculated.

The left side of this equation is only relevant when **y(i)** is real i.e., 1 in this case. Whereas, the right side of the equation is valid when **y(i)** is fake i.e., 0 in this case.

<br>
<p align = "justify">
For a good prediction of the real value (almost close to 1) the left part of the equation becomes 0 due to the log. Whereas, for a bad prediction (close to 0) the left part of the equation returns a high negative value.</p>
<p align = "justify">
Now for the right side of the equation, for a good prediction when we receive a value close to 0 (since the discriminator detected that the sample is fake) the final output becomes 0 as well due to the log. And for a bad prediction when a value close to 1 comes (when the discriminator failed to detect the fake image), the right hand side of the equation, also returns a high negative value.</p>
<br>

> A good predictions returns 0 every time and a bad prediction returns a high negative value. But the negative average over m turns them into positive.

<br>

**Discriminator Phase**

<br>
<p align = "justify">
Initially, the discriminator performs poorly as well, as a result of which its essential to train it as well to update its parameters and make its detection stricter. For this reason, the ouputs of the discriminator is compared with actual features and the overall cost function is calculated which helps to update the parameters accordingly.</p>
<p align = "justify">
In other words, both the real and fake sample are passed through the discriminator of the GAN to compute an output prediction without telling which samples are real or fake. The predicted output is compared with the Binary Cross Entropy (BCE) labels for the predicted category (*real* or *fake* - usually, 1 is used to represent a real, while 0 represents a fake). Finally, after the computation of all these steps, the parameters of the discriminator can be updated accordingly.</p>
<br>

**Generator Phase**

<br>
<p align = "justify">
The generator computes only on the real samples. No fake samples are passed to the generator. The cost function after the calculation is evaluated and the parameters of the generator are updated. In this training phase, only the parameters of the generator are updated accordingly, while the discriminator parameters are neglected.</p>
<p align = "justify">
When the BCE values are equal to real (or 1), only then are the generated parameters updated. Both the generator and discriminator are trained one at a time, and they are both trained alternately. The generator learns from getting feedback if its classification was right or wrong.</p>
<p align = "justify">
Both the discriminator and the generator are trained together. They are usually kept at similar skill levels from the beginning, because a superior discriminator becomes so much better at distinguishing between fake and real that the generator fails to keep up ad produce convincing real samples. A similar case is observed in case of generators, where the discriminator fails to keep up and allows noisy random images to pass as real.</p>
<br>

### Insight on the DCGANs

<br>
<p align = "justify">
This paper provides an insight on creating more stable GANs as they are known to be unstable to train, often resulting in nonsensical outputs. It provides a way to build the GAN in such a way that parts of the generator and the discriminator can be reused for feature extraction. Since GANs provide an alternative to maximum likelihood techniques and are attractive to representation learning, due to the lack of a heuristic cost function, this paper aims to stabilize the GANs for representation learning from unlabeled data.</p>
<br>

**Architecture of the Discriminator**

<br>
<p align = "justify">
The architecture of the discriminator in the DCGAN paper uses, strided convolutions for reducing the dimensions of the feature-vectors rather than any pooling layers and apply a series of leaky_relu, dropout and Batch Normalization (BN) for all layers to stabilize the learning. BN is dropped for input layer and last layer (for the purpose of feature matching). In the end, Global Average Pooling is performed to take the average over the spatial dimensions of the feature vectors. This squashes the tensor dimensions to a single value. After flattening the features, a dense layer of multiple classes is added with softmax activation for multi-class output.</p>
<br>

```python
def discriminator(x, dropout_rate = 0., is_training = True, reuse = False):
    # input x -> n+1 classes
    with tf.variable_scope('Discriminator', reuse = reuse): 
      # x = ?*64*64*1
      
      #Layer 1
      conv1 = tf.layers.conv2d(x, 128, kernel_size = [4,4], strides = [2,2],
                              padding = 'same', activation = tf.nn.leaky_relu, name = 'conv1') # ?*32*32*128
      #No batch-norm for input layer
      dropout1 = tf.nn.dropout(conv1, dropout_rate)
      
      #Layer2
      conv2 = tf.layers.conv2d(dropout1, 256, kernel_size = [4,4], strides = [2,2],
                              padding = 'same', activation = tf.nn.leaky_relu, name = 'conv2') # ?*16*16*256
      batch2 = tf.layers.batch_normalization(conv2, training = is_training)
      dropout2 = tf.nn.dropout(batch2, dropout_rate)
      
      #Layer3
      conv3 = tf.layers.conv2d(dropout2, 512, kernel_size = [4,4], strides = [4,4],
                              padding = 'same', activation = tf.nn.leaky_relu, name = 'conv3') # ?*4*4*512
      batch3 = tf.layers.batch_normalization(conv3, training = is_training)
      dropout3 = tf.nn.dropout(batch3, dropout_rate)
        
      # Layer 4
      conv4 = tf.layers.conv2d(dropout3, 1024, kernel_size=[3,3], strides=[1,1],
                               padding='valid',activation = tf.nn.leaky_relu, name='conv4') # ?*2*2*1024
      # No batch-norm as this layer's op will be used in feature matching loss
      # No dropout as feature matching needs to be definite on logits

      # Layer 5
      # Note: Applying Global average pooling        
      flatten = tf.reduce_mean(conv4, axis = [1,2])
      logits_D = tf.layers.dense(flatten, (1 + num_classes))
      out_D = tf.nn.softmax(logits_D)     
    return flatten,logits_D,out_D
```

<br>

**Architecture of the Generator**

<br>
<p align = "justify">
The generator architecture is designed to mirror the discriminator's spatial outputs. Fractional strided convolutions are used to increase the spatial dimension of the representation. An input of 4-D tensor of noise z is fed which undergoes a series of transposed convolutions, relu, BN(except at output layer) and dropout operations. Finally tanh activation maps the output image in range (-1,1).</p>

<br>

```python
def generator(z, dropout_rate = 0., is_training = True, reuse = False):
    # input latent z -> image x
    with tf.variable_scope('Generator', reuse = reuse):
      #Layer 1
      deconv1 = tf.layers.conv2d_transpose(z, 512, kernel_size = [4,4],
                                         strides = [1,1], padding = 'valid',
                                        activation = tf.nn.relu, name = 'deconv1') # ?*4*4*512
      batch1 = tf.layers.batch_normalization(deconv1, training = is_training)
      dropout1 = tf.nn.dropout(batch1, dropout_rate)
      
      #Layer 2
      deconv2 = tf.layers.conv2d_transpose(dropout1, 256, kernel_size = [4,4],
                                         strides = [4,4], padding = 'same',
                                        activation = tf.nn.relu, name = 'deconv2')# ?*16*16*256
      batch2 = tf.layers.batch_normalization(deconv2, training = is_training)
      dropout2 = tf.nn.dropout(batch2, dropout_rate)
        
      #Layer 3
      deconv3 = tf.layers.conv2d_transpose(dropout2, 128, kernel_size = [4,4],
                                         strides = [2,2], padding = 'same',
                                        activation = tf.nn.relu, name = 'deconv3')# ?*32*32*256
      batch3 = tf.layers.batch_normalization(deconv3, training = is_training)
      dropout3 = tf.nn.dropout(batch3, dropout_rate)
      
      #Output layer
      deconv4 = tf.layers.conv2d_transpose(dropout3, 1, kernel_size = [4,4],
                                        strides = [2,2], padding = 'same',
                                        activation = None, name = 'deconv4')# ?*64*64*1
      out = tf.nn.tanh(deconv4)
    return out
```

<br>

**Loss Evaluation and Training**

<br>
<p align = "justify">
The discriminator loss for unlabeled data can be thought of as a binary sigmoid loss by asserting R/F neuron output to 1 for fake images and 0 for real images. Whereas, generator loss is a combination of fake_image loss which falsely wants to assert R/F neuron output to 0 and feature matching loss which penalizes the mean absolute error between the average value of some set of features on the training data and the average values of that set of features on the generated samples.</p>
<p align = "justify">
The training images are resized from [batch_size, 28, 28, 1] to [batch_size, 64, 64, 1] to fit the generator/discriminator architectures. Losses, accuracies and generated samples are calculated and are observed to improve over each epoch.</p>
<br>

> [Semi-Supervised Learning and GANs](https://towardsdatascience.com/semi-supervised-learning-and-gans-f23bbf4ac683)

This contains a detailed explanation of the loss evaluation and the training procedure of the DCGANs.

<br>

**Architecture Guidelines for Stable Deep Convolutional GANs:**

<br>

<div align = justify>
  <ul>
    <li>Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generaror) The generator learns its own spatial downsampling itself using convolution while the discriminator learns its own spatial upsampling.</li>
    <br>
    <li>Use batchnorm to both the generator and the discriminator.  BN sttabilizes learnig by normalizing the input to each unit to have zero mean and unit variance but directly applying BN to all layers results in sample oscillations and model instability. So BN is avoided in generator output layer and discriminator input layer.</li>
    <br>
    <li>Remove any fully connected hidden layers for deeper architectures.  The first layer of the generator and the last layer of the discriminator may use fully connected hidden layers but there are no fully connected layer for all hidden layers.</li>
    <br>
    <li>Use ReLU activation in generator for all layers except for the ouput, which will use Tanh.</li>
    <br>
    <li>Use LeakyReLU activation in the discriminator for all layers.</li>
  </ul>
</div>
<br>

### Insight on Improved Techniques for Training GANs

<br>
<p align = "justify">
Generative Adversarial Networks (GANs) are a class for learning generative models based on game theory. </p>

- The goal of GANs is to train a generator network  $G(z; \theta^{(G)})$ that produces samples from the data distribution $p_{data}(x)$, transforming vectors of noise z as $x = G(z; \theta^{(G)})$.
- The training signal for G is provided by a discriminator network $D(x)$ that is trained to distinguish samples from the generator distribution $p_{model}(x)$ from real data. The generator network G in turn is trained to fool the discriminator into accepting its output as being real.

<br>
<p align = "justify">
Although recent applications of GANs can produce excellent samples, GANs are typically trained using gradient descent techniques that are designed to find a low value of a cost function. However, training GANs must require a Nash equilibrium of a non-convex game with continuous, high-dimensional patterns. In this paper, several techniques are shown which encourage converge of the GANs game.</p>

**Convergence of GANs**

Training GANs consists of finding a Nash equilibrium to a two-player non-cooperative game, where each player wishes to minimize their own cost function. $J^{(D)}(\theta^{(D)}, \theta^{(G)})$ for the discriminator and $J^{(G)}(\theta^{(D)}, \theta^{(G)})$ for the generator.

<br>

> A Nash equilibrium is a point $(\theta^{(D)}, \theta^{(G)})$ such that  $J^{(D)}$ is at a minimum with respect to $\theta^{(D)}$ and $J^{(G)}$ is at a minimum with respect to $\theta^{(G)}$

<br>

The idea that a Nash equilibrium occurs when each player has minimal cost seems to intuitively motivate the idea of using traditional gradient-based minimization techniques to minimize each player’s cost simultaneously. Unfortunately a modition to $\theta^{(D)}$ that reduces $J^{(D)}$ can increase $J^{(G)}$, and a modification to $\theta^{(G)}$ that reduces $J^{(G)}$ can increase $J^{(D)}$. Previous approaches to GAN training have thus applied gradient descent on each player’s cost simulatneously, despite the lack of guarantee that this procedure will converge.

<br>

The following techniques are heuristically motivated to encourage convergence,

<br>

**Feature Matching**

<br>
<p align = "justify">
Feature matching addresses the instability of GANs by specifying a new objective for the generator that prevents it from overtraining on the current discriminator. Instead of directly maximizing the output of the discriminator, the new objective requires the generator to generate data that matches the statistics of the real data, where we use the discriminator only to specify the statistics that we think are worth matching. Specifically, we train the generator to match the expected value of the features on an intermediate layer of the discriminator. This is a natural choice of statistics for the generator to match, since by training the discriminator we ask it to find those features that are most discriminative of real data versus data generated by the current model.</p>
<p align = "justify">
Let f(x) denote activations on an intermediate layer of the discriminator, then the new objective function for generator becomes,</p>
<br>

$$
||\mathbb{E}_{x\sim{p_{data}}}f(x) - \mathbb{E}_{x\sim{p_{z(z)}}}f(G(z))||_2^2
$$

<br>
<p align = "justify">
The discriminator, and hence f(x), are trained in the usual way. As with regular GAN training, the objective has a fixed point where G exactly matches the distribution of training data.</p>

<br>
<div align = "center"> <img src = "Assets/Improved Discriminator.jpeg" width = 300> </div>
<br>

> The means of the real image features are computed per minibatch which fluctuate on every batch. It is good news in mitigating the mode collapse. It introduces randomness that makes the discriminator harder to overfit itself.

<br>

**Minibatch Discrimination**

<br>
<p align = "justify">
One of the main reason for mode collapse in GAN is for the generator to collapse to a parameter setting where it always emits the same point. When collaping to a single mode, the gradient of the discriminator points in similar direction for many similar points. Because the discriminator processes each example independently, there is no coordination between its gradients and thus no mechanism to guide the outputs of the generator to be more dissimilar. Instead all the outputs race to the same single point which the discriminator believes is highly realistic.</p>
<p align = "justify">
After collapse has occurred, the discriminator learns that this single point comes from the generator, but gradient descent is unable to separate the identical outputs. The gradients of the discriminator then push the single point produced by the generator around space forever, and the algorithm cannot converge to a distribution with the correct amount of entropy. The strategy to avoid this type of failure is to allow the discriminator to look at multiple data examples in combination and perform minibatch discrimination.</p>

<br>
<div align = "center"> <img src = "Assets/Improved Minibatch Discrimination.jpeg" width = 300> </div>
<br>

<p align = "justify">
The concept of minibatch discrimination is quite general: any discriminator model that looks at multiple examples in combination, rather than in isolation, could potentially help avoid collapse of the generator. If the mode starts to collapse, the similarity of generated images increases. The discriminator can use this score to detect generated images and penalize the generator if mode is collapsing.</p>

The similarity **o(xi)** between the image **xi** and other images in the same batch is computed by a transformation matrix **T.** In the figure below, **xi** is the input image and **xj** is the rest of the images in the same batch. 

<br>
<div align = "center"> <img src = "Assets/Minibatch Discrimination Procedure.jpeg" width = 300> </div>
<br>

We use a transformation matrix **T** to transform the features **xi** to **Mi** which is a B×C matrix.  We then derive the similarity **c(xi, xj)** between image i and j using the L-1 norm before calulating the similarity between **xi** and the rest of the images in the batch.

<br>
<div align = "center"> <img src = "Assets/MD Equations.png" width = 300> </div>
<br>

> “*Minibatch discrimination allows us to generate visually appealing samples very quickly, and in this regard it is superior to feature matching.” ~ Improved Techniques for Training GANs*

<br>

**Historial Averaging**

<br>
<p align = "justify">
In historial average, the model paramters for the last <b>t</b> models are tracked. Alternatively, a running average of the model parameters are kept for long sequence of models.</p>

Finally an **L2** cost is added to the cost function to penalize the model,

<br>

$$
||\theta - \frac1t\Sigma_{i=1}^t\theta[i]||_2
$$

<br>

where $\theta[i]$ is the value of the parameters at past time **i.**

<br>

> For GANs with non-convex object function, historical averaging may stop models circle around the equilibrium point and act as a damping force to converge the model.

<br>

**One-sided Label Smoothing**

<br>
<p align = "justify">
Label smoothing, replaces the 0 and 1 targets for a classifier with smoothed values, like .9 or .1, to reduce the vulnerability of neural networks to adversarial examples.</p>
<p align = "justify">
Replacing positive classification targets with α and negative targets with β, the optimal discriminator becomes,</p>

<br>

$$
 D(x) = \frac{αp_{data}(x)+βp_{model}(x)}{p_{data}(x)+p_{model}(x)}
$$

<br>

The presence of pmodel in the numerator is problematic because, in areas where $p_{data}$ is approximately zero and $p_{model}$model is large, erroneous samples from $p_{model}$ have no incentive to move nearer to the data. Therefore only the positive labels to α are smoothed, leaving negative labels set to 0.

<br>

> To avoid the problem of overconfidence, we penalize the discriminator when the prediction for any real images go beyond 0.9 (*D(real image)>0.9*). This is done by setting the target label value to be 0.9 instead of 1.0.

<br>

**Virtual Batch Normalization**

<br>
<p align = "justify">
Batch normalization greatly improves optimization of neural networks, and was shown to be highly effective for DCGANs.  However, it causes the output of a neural network for an input example x to be highly dependent on several other inputs x’ in the same minibatch.</p>
<p align = "justify">
To avoid this problem virtual batch normalization (VBN) is used, in which each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training, and on x itself. The reference batch is normalized using only its own statistics.</p>
<br>

> VBN is computationally expensive because it requires running forward propagation on two minibatches of data, so it is only used in the generator network.

<br>

### **Code Snippet for GANs**

<br>

The following code contains the construction for a basic Generative Adversarial Network, which generates a specific kind of matrices. 

<br>

>[GANs for Generating Weakly-Reflexive Relations](https://www.kaggle.com/code/samyabose/gans-for-generating-weakly-reflexive-relations?kernelSessionId=97919755)

It contains a detailed overview of the networks (both Generator and Discriminator) along with that a detailed walkthrough of the updation of their weights and loss functions.

<br>

**Keras Implementation**

[Keras-GAN/cgan.py at master · eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py)

<br>

**TensorFlow Implementation**

[An introduction to Generative Adversarial Networks (with code in TensorFlow)](https://aylien.com/blog/introduction-generative-adversarial-networks-code-tensorflow)

<br>

**PyTorch Implementation**

[Complete Guide to Generative Adversarial Networks (GANs)](https://blog.paperspace.com/complete-guide-to-gans/)

<br>

---

<br>

<img src = Assets/Logo.png width = 50>

<br> 

**Project Name**

Automated Text-Translation and Data Visualization using Generative Adversarial Networks (GANs)

@eYSIP-2022

<br>

**Links**

Cover Photo ~ [Photo by Oliver Pecker on Unsplash](https://unsplash.com/photos/HONJP8DyiSM?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)

Logo ~ [WEB Free Fonts for Windows and Mac / Font free Download - OnlineWebFonts.COM](http://www.onlinewebfonts.com)

Icon licensed by CC 3.0

<br>

**Notes**

CNN: [Convolutional Neural Networks (CNNs)](https://samya-ravenxi.notion.site/Convolutional-Neural-Networks-CNNs-cb2ca9e7765b46f4a6437af0540f7abd)

RNN: [Recurrent Neural Networks (RNNs)](https://samya-ravenxi.notion.site/Recurrent-Neural-Networks-RNNs-e904a4e282bf4141865204a47e01521f)

LSTM: [Long Short-Term Memory Networks (LSTMs)](https://samya-ravenxi.notion.site/Long-Short-Term-Memory-Networks-LSTMs-5f1ffa44545641dfbe4c7e54ca85c6b4)

Attention: [Attention Models](https://samya-ravenxi.notion.site/Attention-Models-6f1b3e2c50fe4ab38264db9b01f6c578)

Transformer: [Transformers](https://samya-ravenxi.notion.site/Transformers-b31498d91b7c4198a29aaae5059e674f)

GAN: [Generative Adversarial Networks (GANs)](https://samya-ravenxi.notion.site/Generative-Adversarial-Networks-GANs-5a6b5dae6268418f988762f23c4996b3)

<br>

**References**

[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434v2)

[Complete Guide to Generative Adversarial Networks (GANs)](https://blog.paperspace.com/complete-guide-to-gans/)

[A Gentle Introduction to Generative Adversarial Networks (GANs) - Machine Learning Mastery](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)


[Generative adversarial networks: What GANs are and how they've evolved](https://venturebeat.com/2019/12/26/gan-generative-adversarial-network-explainer-ai-machine-learning/)

[Generative Adversarial Networks (GANs)](https://www.coursera.org/specializations/generative-adversarial-networks-gans)

[Semi-Supervised Learning and GANs](https://towardsdatascience.com/semi-supervised-learning-and-gans-f23bbf4ac683)

[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)

[GAN — Ways to improve GAN performance](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b#:~:text=%20GAN%20%E2%80%94%20Ways%20to%20improve%20GAN%20performance,last%20t%20models.%205%20Reference.%20%20More%20)