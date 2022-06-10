<img src = "Assets/Cover.jpg">

<br>

# Recurrent Neural Networks (RNNs)

<br>

<p align = justify>
A glaring limitation of Vanilla Neural Networks is that their API is too constrained: they accept a fixed-sized vector as input (e.g. an image) and produce a fixed-sized vector as output (e.g. probabilities of different classes). These models perform this mapping using a fixed amount of computational steps (e.g. the number of layers in the model).
</p>

<p align = justify>
Whereas, the core reason that recurrent nets are more exciting is that they allow us to operate over <i>sequences</i> of vectors: Sequences in the input, the output, or in the most general case both.
</p>

<br>

<div align = center><img src = "Assets/RNN - Layouts.jpeg" width = 400></div>

<br>

<p align = justify>
🧾 From left to right: <b>(1)</b> Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). <b>(2)</b> Sequence output (e.g. image captioning takes an image and outputs a sentence of words). <b>(3)</b> Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). <b>(4)</b> Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). <b>(5)</b> Synced sequence input and output (e.g. video classification where we wish to label each frame of the video).
</p>
<br>

## Computation using RNN:

<br>

<p align = justify>
At the core, RNNs have a deceptively simple API: They accept an input vector `x` and give you an output vector `y`. However, crucially this output vector's contents are influenced not only by the input you just fed in, but also on the entire history of inputs you've fed in in the past. Written as a class, the RNN's API consists of a single `step`function: </p>

<br>

```python
rnn = RNN()
y = rnn.step(x) # x is an input vector, y is the RNN's output vector
```
<br>

<p align = justify>
The RNN class has some internal state that it gets to update every time `step` is called. In the simplest case this state consists of a single hidden vector `h`. Here is an implementation of the step function in a Vanilla RNN:</p>

<br>

```python
class RNN:
  # ...
  def step(self, x):
    # update the hidden state
    self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
    # compute the output vector
    y = np.dot(self.W_hy, self.h)
    return y
```
<br>

The above specifies the forward pass of a vanilla RNN. 

<p align = justify>
This RNN's parameters are the three matrices 'W_hh, W_xh, W_hy'. The hidden state 'self.h' is initialized with the zero vector. The `np.tanh` function implements a non-linearity that squashes the activations to the range '[-1, 1]'. There are two terms inside of the tanh: one is based on the previous hidden state and one is based on the current input. In numpy 'np.dot' is matrix multiplication. The two intermediates interact with addition, and then get squished by the tanh into the new state vector.</p>

$$h_t = tanh(W_h{}_hh_t{}_-{}_1 + W_x{}_hx_t)$$ 

where tanh is applied elementwise.

<br>

<div align = center><img src="Assets/Unfolded RNN.webp" width = 400></div>

<br>

2-layer recurrent network:

<br>

```python
y1 = rnn1.step(x)
y = rnn2.step(y1)
```

<br>

<p align = justify>
Here there are two separate RNNs: One RNN is receiving the input vectors and the second RNN is receiving the output of the first RNN as its input.</p>
<br>

### Limitations of RNN:

<br>

**Vanishing Gradient -** 

<p align = justify>
The vanishing gradient problem occurs when the backpropagation algorithm moves back through all of the neurons of the neural net to update their weights. The nature of recurrent neural networks means that the cost function computed at a deep layer of the neural net will be used to change the weights of neurons at shallower layers. The mathematics that computes this change is multiplicative, which means that the gradient calculated in a step that is deep in the neural network will be multiplied back through the weights earlier in the network. As a result, gradient calculated deep in the network is "diluted" as it moves back through the net, which can cause the gradient to vanish.</p>
<br>

**Exploding Gradient -** 

<p align = justify>
The exploding gradient problem occurs when the backpropagation algorithm moves back through all of the neurons of the neural net to update their weights. The nature of recurrent neural networks means that the cost function computed at a deep layer of the neural net will be used to change the weights of neurons at shallower layers. The mathematics that computes this change is multiplicative, which means that the gradient calculated in a step that is deep in the neural network will be multiplied back through the weights earlier in the network. As a result, gradient calculated deep in the network is "concentrated" as it moves back through the net, which can cause the gradient to explode.</p>
<br>

**Long-Term Dependencies -** 

<p align = justify>
One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, such as using previous video frames might inform the understanding of the present frame. Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in "the clouds are in the <b>sky</b>" we don't need any further context - it's pretty obvious the next word is going to be sky. In such cases, where the gap between the relevant information and the place that it's needed is small, RNNs can learn to use the past information.</p>
<p align = justify>
But there are also cases where we need more context. Consider trying to predict the last word in the text "I grew up in France... I speak fluent <b>French</b>." Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It's entirely possible for the gap between the relevant information and the point where it is needed to become very large.</p>

<br>

Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

**Solving Vanishing and Exploding Gradients -** 

<p align = justify>
- For exploding gradients, it is possible to use a modified version of the backpropagation algorithm called <b>truncated backpropagation<b>. The truncated backpropagation algorithm limits that number of timesteps that the backpropagation will be performed on, stopping the algorithm before the exploding gradient problem occurs.</p>
<br>
<p align = justify>
- Weight initialization is one technique that can be used to solve the vanishing gradient problem. It involves artificially creating an initial value for weights in a neural network to prevent the backpropagation algorithm from assigning weights that are unrealistically small.</p>
<br>
<p align = justify>
- The most important solution to the vanishing gradient problem is a specific type of neural network called Long Short-Term Memory Networks (LSTMs), which were pioneered by <b>Sepp Hochreiter</b> and <b>Jürgen Schmidhuber</b>.

<br>

### **Code Snippet for RNN Architecture:**

<br>

[RNN From Scratch | Building RNN Model In Python](https://www.analyticsvidhya.com/blog/2019/01/fundamentals-deep-learning-recurrent-neural-networks-scratch-python/)

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

[The Ultimate Guide to Recurrent Neural Networks in Python](https://www.freecodecamp.org/news/the-ultimate-guide-to-recurrent-neural-networks-in-python/)

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)