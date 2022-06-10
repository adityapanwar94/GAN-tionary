<img src = "Assets/Cover.jpg">

<br>

# Long Short-Term Memory Networks (LSTMs)

<p align = "justify">
Long Short Term Memory networks - usually just called "LSTMs" - are a special kind of RNN, capable of learning long-term dependencies. They were introduced by <a href = "http://www.bioinf.jku.at/publications/older/2604.pdf" alt = "Hochreiter & Schmidhuber (1997)">Hochreiter & Schmidhuber (1997)</a>, and were refined and popularized by many people in following work. <a href = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/#fn1">[1]</a> They work tremendously well on a large variety of problems, and are now widely used.</p>

<p align = "justify">
All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. </p>

<br>

<b>The repeating module in a standard RNN contains a single layer</b>

<br>

<div align = center><img src = "Assets/RNN ULSTM.png" width = 400></div>

<br>

<div align = center>The repeating module in a standard RNN contains a single layer</div>

<br>
<br>

<p align = justify>
LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.</p>

<br>

<div align = center><img src = "Assets/LSTM.png" width = 400></div>

<br>

<div align = center> The repeating module in an LSTM contains four interacting layers</div>

<br>
<br>

## LSTM Architecture:

<br>

- **Forget Gate Layer -**

<p align = "justify">
The first step in LSTM is to decide what information is going to be thrown away from the cell state. This decision is made by a sigmoid layer called the "forget gate layer."</p>

It looks at $h_t{}_-{}_1$ and $x_t$, and outputs a number between *0* and  *1* for each number in the cell state $C_t{}_-{}_1$. *1* represents "completely keep this" while *0* represents "completely get rid of this."

<br>
<div align = center><img src = "Assets/Forget Gate.png" width = 400></div>
<br>

- **Input Gate Layer -**

<br>
<p align = "justify">
The next step is to decide what new information is going to be stored in the cell state. This has two parts. First, a sigmoid layer called the "input gate layer" decides which values will get updated. Next, a tanh layer creates a vector of new candidate values,</p>

$C^-_t$, that could be added to the state. In the next step, these two are combined to create an update to the state.

<br>
<div align = center><img src = "Assets/Input Gate.png" width = 400></div>
<br>

- **Cell State -**

In the next step to update the old cell state, $C_t{}_-{}_1$, into the new cell state $C_t$, multiply the old state is multiplied by $f_t$, forgetting the things we decided to forget earlier. Then $i_t * C^-_t$ is added to it. This is the new candidate value, scaled by how much we decided to update each state value.

<br>
<div align = center><img src = "Assets/Cell State.png" width = 400></div>
<br>

- **Output Gate Layer -**

<p align = justify>
Finally, in this step the output is formed. This output will be based on the cell state, but will be a filtered version. First, sigmoid layer is used which decides what parts of the cell state is going to output. Then, cell state passes through tanh, to push the values to be between -1 and 1) and multiply it by the output of the sigmoid gate, so that the output will only contain the parts we decided to.</p>

<br>
<div align = center><img src = "Assets/Output Gate.png" width = 400></div>
<br>

### Limitations of LSTM:

<br>
<div align = justify>
       <ul>  
              <li>LSTMs became popular because they could solve the problem of vanishing gradients. But it turns out, they fail to remove it completely. The problem lies in the fact that the data still has to move from cell to cell for its evaluation. Moreover, the cell has become quite complex now with the additional features (such as forget gates) being brought into the picture.</li>
              <br>
              <li>LSTMs became popular because they could solve the problem of vanishing gradients. But it turns out, they fail to remove it completely. The problem lies in the fact that the data still has to move from cell to cell for its evaluation. Moreover, the cell has become quite complex now with the additional features (such as forget gates) being brought into the picture.</li>
              <br>
              <li>They require a lot of resources and time to get trained and become ready for real-world applications. In technical terms, they need high memory-bandwidth because of linear layers present in each cell which the system usually fails to provide for. Thus, hardware-wise, LSTMs become quite inefficient.</li>
              <br>
              <li>With the rise of data mining, developers are looking for a model that can remember past information for a longer time than LSTMs. The source of inspiration for such kind of model is the human habit of dividing a given piece of information into small parts for easy remembrance.</li>
              <br>
              <li>LSTMs get affected by different random weight initialization and hence behave quite similar to that of a feed-forward neural net. They prefer small weight initialization instead.</li>
              <br>
              <li>LSTMs are prone to overfitting and it is difficult to apply the dropout algorithm to curb this issue. Dropout is a regularization method where input and recurrent connections to LSTM units are probabilistically excluded from activation and weight updates while training a network.</li>
       </ul>
</div>
<br>

### **Code Snippet for LSTM Architecture:**

<br>

```python
model = Sequential() 
model.add(Embedding(num_words, 50, input_length=200)) 
model.add(Dropout(0.2)) 
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(250, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

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

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


[Understanding of LSTM Networks - GeeksforGeeks](https://www.geeksforgeeks.org/understanding-of-lstm-networks/#:~:text=%20As%20it%20is%20said%2C%20everything%20in%20this,past%20information%20for%20a%20longer%20time...%20More%20)

[Two Ways to Implement LSTM Network using Python - with TensorFlow and Keras | Rubik's Code](https://rubikscode.net/2018/03/26/two-ways-to-implement-lstm-network-using-python-with-tensorflow-and-keras/#:~:text=%20Two%20Ways%20to%20Implement%20LSTM%20Network%20using,a%20lot%20of%20things%20we%20had...%20More%20)
