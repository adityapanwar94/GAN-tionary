# PReFIL 

## Parallel Recurrent fusion of Image and Language

PReFIL learns bimodal embeddings by fusing question and image features and then intelligently
aggregates these learned embeddings to answer the given question.

<aside>
💡 PReFIL greatly surpasses state-of-the art systems and human baselines on both the
FigureQA and DVQA datasets.

</aside>

The PReFIL architecture is relatively simple without employing specialized relational or attention modules.

### Datasets for testing PReFIL

- DVQA has over 3 million question answer pairs for
300,000 images for bar charts.

![Untitled](https://user-images.githubusercontent.com/72121513/180183824-0f640112-483c-42ea-ac62-5f4d16fb812a.png)


**Limitations of the datasets**

All of DVQA’s charts were made with Matplotlib and all of FigureQA’s
were made with Bokeh. 

The variation introduced is limited to the capabilities of these packages. 

FigureQA uses only generic titles and other chart elements.

---

<aside>
💡 Compared to existing work, PReFIL does not employ complex attention or relational modules, and unlike FigureNet, it does not require additional supervised annotations for training on FigureQA.

</aside>

---

## PReFIL model

![Untitled_2](https://user-images.githubusercontent.com/72121513/180183748-aaacb4e5-5c61-4091-9ed8-5dde49068136.png)

- PReFIL consists of two parallel Q+I fusion branches.
- Each branch takes in question features (from an LSTM) and image features from two locations of a 40-layer DenseNet.
- Low-level features (from layer 14) and High level features (from layer 40).
- Each Q+I fusion block concatenates the question features to each element of the convolutional feature map.
- Then it has a series of 1x1 convolutions to create question-specific bimodal embeddings.
- These embeddings are then recurrently aggregated and finally fed to a classifier that predicts the answer.
- For DVQA, an additional fourth OCRintegration component is required

---

## Multistage Image Encoder

For all model variants, image encoder is a DenseNet trained from scratch. 
**DenseNet** is an efficient architecture for training deep convolutional neural networks. 

Comprised of several ‘dense’ blocks and ‘transition’ blocks between the dense blocks.(A dense block has several Convolutional network)

The Transition block sits between two dense blocks and serves to change featuremap sizes via convolution and pooling

<aside>
💡 In data visualizations, simpler features such as color patches, lines, textures, etc. convey important information that is often abstracted
away by deeper layers of a CNN. Hence, we use both low and high-level convolutional features in our model, both of
which are fed to parallel fusion module alongside question embeddings learned using an LSTM

</aside>

---

## Parallel Fusion of Image and Language

Jointly modulating visual features using vision and language features can allow models to learn richer features for downstream tasks

*Q+I fusion block does this by first concatenating all of the input convolutional feature
map’s spatial locations with the question features, and then bimodal fusion occurs using a series of layers that use 1x1 convolutions*

---

## Recurrent Aggregation of bimodal features

Model aggregates information using a bidirectional gated recurrent unit

The aggregated features are sent to a classifier to predict the answer.

---

## OCR Integration for DVQA dataset

Unlike FigureQA and most other VQA tasks, DVQA requires OCR to answer its reasoning and data questions.

To integrate OCR into PReFIL, we use the same dynamic encoding scheme used by the SANDY model for DVQA.

To assess impact of OCR, we test three OCR versions as well as a version of algorithm trained without the dynamic encoding

---

## Model and Training Hyperparameters

**Question Encoding:** Question words are represented by 32 dimensional learned word embedding and passed
through an LSTM which provides a 256-dimensional embedding representing the whole question.

**DenseNet:** We use a 40 layer DenseNet composed of 3 dense blocks with 12 layers each. The number of initial
filters is 64 and the growth rate is set to 32.

**Preprocessing:** DVQA images are resized to a size of 256x256. FigureQA images are all differently sized but we
resize them to 320x224 which maintains an average widthheight aspect ratio.

**Q+I Fusion:** Inputs to Q+I block are batchnormed. Each Q+I fusion block is composed of four 1  1 convolutions with 256 channels and ReLU.

**Recurrent Fusion:** The bimodal features are aggregated using a 256 dimensional bi-directional GRU. The forward
and backward direction outputs are concatenated to form a 512 dimensional vector which is fed to the classifier.

**Classifier:** The aggregated bimodal features are projectedvto a 1024 fully connected ReLU layer, which was
regularized using dropout of 0.5 during training. 

The classification layer is binary for FigureQA. 

*For DVQA, the classification layer has 107 units, with 77 units for predicting
‘common’ answers such as ‘yes’, ‘no’, ‘three groups’, etc, and 30 special tokens for predicting answers that require
OCR, which allows PReFIL to produce OOV answer tokens that are unseen during training.*

---

## Results

### DVQA

<aside>
💡 PReFIL surpasses SANDY by over 40% in accuracy when both the baseline SANDY and our PReFIL method have access
to a perfect Oracle OCR. When using Tesseract OCR, we obtain about a 24%
improvement overall on both test sets.

</aside>

***Across all OCR variants, PReFIL outperforms SANDY.
Moreover, PReFIL’s performance scales much better when better OCR is available: 11% gain for SANDY vs. 26%
gain for PReFIL when moving from the imperfect Tesseract OCR setup to the perfect Oracle OCR setup.***

---

## Resources/ References:

[Answering Questions about Data Visualizations using Efficient Bimodal Fusion](https://arxiv.org/abs/1908.01801)

[https://github.com/kushalkafle/PREFIL](https://github.com/kushalkafle/PREFIL)
