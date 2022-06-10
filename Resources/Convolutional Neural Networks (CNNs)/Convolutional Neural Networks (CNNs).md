<img src = "Assets/Cover.jpg">

<br>

# Convolutional Neural Networks (CNNs)

<br>

<p align = "justify">
Convolutional Neural Networks, take advantage of the fact that, the input consists of images and they constrain the architecture in a more sensible way, unlike a regular Neural Network. The layers of a CNN typically have neurons arranged in 3 dimensions: width, height and depth.
</p>

<br>
<div align = "center"> <img src = "Assets/CNN - Structure.jpeg" width = 300> </div>
<br>

<p align = "justify">
🧾 Left: A regular 3-layer Neural Network. Right: A CNN arranges its neurons in three dimensions (width, height, depth), as visualized in one of the layers. Every layer of a CNN transforms the 3D input volume to a 3D output volume of neuron activations. In this example, the red input layer holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels).
</p>

<br>

## Layers used to build CNN

<br>

<p align = "justify">
A simple CNN is a sequence of layers where every layer transforms one volume of activations to another through a differential function. Three main types of layers are typically used to build CNN architectures - <b>Convolutional Layer</b>, <b>Pooling Layer</b> and <b>Fully-Connected Layer</b>.
<p>

<br>

### Convolutional Layer

<br>

<p align = "justify">
Convolutional Layer computes the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region, that they are connected to in the input volume.
</p>

<p align = justify>
Convolutional Layer’s parameters consist of a set of learnable filters, where every filter is small spatially (along width and height), but extends through the full depth of the input volume. During the forward pass, each filter is moved across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As the filter moves over the spatial features of the input volume, a 2-dimensional activation map is generated that gives the responses of that filter at every spatial position. Intuitively, the network learns filters that activate when it encounters some type of visual feature and the entire set of filters in each Convolutional Layer and each of them generated a separate 2-dimensional activation map. These activation maps are stacked along the depth dimension to produce the output volume.
</p>

<br>

<p align = "justify">
A Relu Layer is often applied after the convolutional layer, it reduce the non-linearity of the input volume as it uses $max(0,x)$ and sets the thresholding at zero.
</p>
<br>

- **Local Connectivity**: <p align = justify>When dealing with high-dimensional inputs such as images, it is impractical to connect neurons to all neurons in the previous volume. Instead, each neuron is connected to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron (equivalently this is the filter size). The extent of the connectivity along the depth axis is always equal to the depth of the input volume.</p>

<br>

- **Spatial Arrangement:** <p align = justify>Three hyperparameters mainly control the size of the output volume: the <b>Depth</b>, <b>Stride</b> and <b>Zero-Padding.</b>
    - **Depth -** <p align = justify>Depth of the output volume is a hyperparameter that corresponds to the number of filters that will be used. Each of these filters are learning to look for something different in the input.</p>
    - **Stride -** <p align = justify>It is the amount by which the filter is shifted. When the stride is $k$, the filter is moved $k$ pixels at a time. As a result, this produces smaller output volumes spatially.</p>

<br>

- **Zero-Padding:** <p align = justify>It is a hyperparameter that is used to pad the input volume with zeros around the border. It is mainly used to control the spatial size of the output volumes.</p>

To calculate the output volume as a function of the Input Volume size $n$, a receptive field size of the Convolutional Layer neurons $f$, the Stride which is applied $s$ and the amount of Zero-Padding used, $p$ on the border:

$$
(n-f+2p)/(s+1)
$$

<br>

**Demo for Convolutional Neural Network:**

<br>

<div align = center><img src = "Assets/CNN - demo.gif" width = 400></div>

<br>

### Pooling Layer

<br>

Pooling Layer performs down sampling operation or dimensionality reduction along the spatial dimensions (height, width).

<p align = justify>
It is standard practice to periodically insert a Pooling Layer in-between successive Convolutional Layer in a CNN architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network and hence to also control overfitting.
</p>
<br>

<div align = center><img src = Assets/Pooling.jpeg width = 250></div>

<br>

### Fully-Connected Layer

<br>

<p align = justify>
Fully-Connected Layer computes the class scores, result in volume [1x1x<b>n</b>], where each of the numbers <b>n</b> corresponds to a class score among <b>n</b> categories. Each neuron in this layer will be connected to all the numbers in the previous volume.</p>

<p align = justify>
Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.</p>

<br>

## Code Snippet for CNN Architecture

<br>

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
```
<br>

### Advantages of using CNN over Traditional Neural Networks:

<br>

<div align = justify>
    <ul>
        <li><b>Weight sharing -</b> It makes use of Local Spatial coherence that provides same weights to some of the edges, In this way, this weight sharing minimizes the cost of computing. This is especially useful when GPU is low power or missing.</li>
        <br>
        <li><b>Parameter Reduction -</b> ANNs have large number of trainable parameters. So, traditional ANN will take a long training time as well as memory. CNN has certain components as a part of the architecture which optimizes the number of trainable parameters.</li>
        <br>
        <li><b>Memory Saving -</b> The reduced number of parameters helps in memory saving.</li>
        <br>
        <li><b>Independent of local variations in Image -</b> Since the convolutional neural network makes use of convolution operation, they are independent of local variations in the image.</li>
        <br>
        <li><b>Equivariance -</b> Equivariance is the property of CNNs and one that can be seen as a specific type of parameter sharing. Conceptually, a function can be considered equivariance if, upon a change in the input, a similar change is reflected in the output. Mathematically, it can be represented as f(g(x)) = g(f(x)). It turns out that convolutions are equivariant to many data transformation operations which helps us to identify, how a particular change in input will affect the output. This helps us to identify any drastic change in the output and retain the reliability of the model.</li>
        <br>
        <li><b>Independent of Transformations -</b> CNNs are much more independent to geometrical transformations like Scaling, Rotation etc.</li>
    </ul>
</div>

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

[Importance of Convolutional Neural Network | ML - GeeksforGeeks](https://www.geeksforgeeks.org/importance-of-convolutional-neural-network-ml/)

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)

[Convolutional Neural Network (CNN) | TensorFlow Core](https://www.tensorflow.org/tutorials/images/cnn)