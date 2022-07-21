# DVQA - Understanding Data Visualizations via Question Answering

## Motivation

- VQA systems typically assume two fixed vocabulary dictionaries:
one for encoding words in questions and one for producing
answers. So VQA systems perform poorly on DVQA
- In DVQA, assuming a fixed vocabulary makes
it impossible to properly process many questions or to
generate answers unique to a bar chart

## Contribution of the paper

1. Describes the DVQA dataset, which contains over 3
million image-question pairs about bar charts.
2. Shows that baseline and state of the art VQA models are incapable in the DVQA
3. Two DVQA systems are proposed
    1. One is an
    end-to-end neural network that can read answers from
    the bar chart.
    2. The second is a model that encodes a bar
    chart’s text using a dynamic local dictionary.

## DVQA : Dataset

The questions in the dataset require the ability to reason about the information within a bar chart .

DVQA contains 3,487,194 total question answer pairs for 300,000 images divided into three major question types.

![Untitled](https://user-images.githubusercontent.com/72121513/180182966-633148de-0ebf-4fd2-84c5-126f5252431d.png)


DVQA contains three types of questions:

1. Structure understanding
2. Data retrieval
3. Reasoning.

---

## Baseline Models

1. YES: This model answers ‘YES’ for all questions,
which is the most common answer in DVQA by a
small margin over ‘NO’.
2. IMG: A question-blind model. Images are encoded
using Resnet using the output of its final convolutional
layer after pooling, and then the answer is predicted
from them by an MLP with one hidden-layer that has
1,024 units and a softmax output layer.
3. QUES: An image-blind model. It uses the LSTM encoder
to embed the question, and then the answer is
predicted by an MLP with one hidden-layer that has
1,024 units and a softmax output layer.
4. IMG+QUES: This is a combination of the QUES and
IMG models. It concatenates the LSTM and CNN embeddings,
and then feeds them to an MLP with one
1024-unit hidden layer and a softmax output layer.
5. SAN-VQA:The Stacked Attention Network for VQA. Own implementation
of SAN is used SAN operates on the last CNN
convolutional feature maps, where it processes this
map attentively using the question embedding from
our LSTM-based scheme.

---

## Multi-Output Model (MOM)

Multi-Output Model (MOM) for DVQA uses a dualnetwork
architecture, where one of its sub-networks is able
to generate chart-specific answers. MOM’s classification
sub-network is responsible for generic answers. The classification sub-network is identical to
the SAN-VQA

MOM’s optical character recognition (OCR) sub-network is responsible
for chart-specific answers that must be read from the
bar chart.

MOM must determine whether to use the classification
sub-network (i.e. SAN-VQA) or the OCR sub-network to
answer a question. To determine this, we train a separate binary
classifier that determines which of the outputs to trust

### MOM

![Untitled_2](https://user-images.githubusercontent.com/72121513/180183022-7d0d19fb-944b-4836-9354-bf61518fd4c9.png)

---

## SANDY: SAN with DYnamic Encoding Model

<aside>
💡 MOM handles chart-specific answers by having a subnetwork
capable of generating unique strings; however, it has no explicit ability to visually read bar chart text and
its LSTM question encoding cannot handle chart-specific
words.

</aside>

SANDY uses a dynamic encoding model (DEM)
that explicitly encodes chart-specific words in the question,
and can directly generate chart-specific answers.

*The DEM is a dynamic local dictionary for chart-specific words. This
dictionary is used for encoding words as well as answers.*

To create a local word dictionary, DEM assumes it has
access to an OCR system that gives it the positions and
strings for all text-areas in a bar chart.

<aside>
💡 We test two versions of SANDY. 
The oracle version directly uses annotations from the DVQA dataset to build a DEM. 
The OCR version uses the output of the open-source Tesseract OCR.

</aside>

---

## Training

All of the classification based systems, except SANDY
and the OCR branch of MOM, use a global answer dictionary
from training set containing 1076 words, so they each
have 1076 output units. 

MOM’s OCR branch contains 27
output units; 1 for each alphabet and and 1 reserved for
blank character. 

Similarly, SANDY’s output layer contains
107 units, with the indices 31 through 107 are reserved
for common answers and indices 0 through 30 are reserved
for the local dictionary.
