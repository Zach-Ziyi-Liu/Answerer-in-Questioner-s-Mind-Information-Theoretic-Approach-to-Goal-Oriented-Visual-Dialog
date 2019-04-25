# AQM Code Implementation

## Setup & Dependencies

As we extended the code of [Das & Kottur et al., 2017], our code is implemented in PyTorch v0.3.1.

1. Install Python 3.6.

2. Install PyTorch v0.3.1, preferably with CUDA - running with GPU acceleration is highly recommended.

3. Get the source code:

   ```
   git clone https://github.com/fLY9636/COMS4995-DeepLearning-AQM.git aqm
   ```
   
4. Install requirements.

   ```
   pip install -r requirements.txt
   ```

## Training

AQM basically uses the same training schemes as suggested in [Das & Kottur et al., 2017]. Therefore, for our experiments, we mainly used the pretrained models [Das & Kottur et al., 2017] provided, and additionally trained Abot for indA (with only the difference of random seed) and Abot for depA. All the necessary checkpoints can be downloaded from [here](https://drive.google.com/file/d/1_rIX3mNbrLhP-xLWUAEWM1pY37apswsq/view?usp=sharing), as explained above. However, if you want to train your own models, here is a brief guide.

Random seed can be set using `-randomSeed <seed>`.


```
python train.py -useGPU -trainMode sl-qbot -qstartFrom qbot_sl_ep60.vd -numEpochs 20
```
change trainMode to train sl-abot or sl-qbot

## Evaluation
###Here provided are the commands to evaluate models 

Evalution via PMR
```
python evaluate.py -useGPU -expLowerLimit 0 -expUpperLimit 500 -evalMode AQMBotRank -startFrom abot_ep_15.vd -aqmQStartFrom qbot_ep_15.vd -aqmAStartFrom abot_ep_15.vd
```
Evalution via dialogues
```
python evaluate.py -useGPU -expLowerLimit 0 -expUpperLimit 500 -evalMode AQMdialog -startFrom abot_ep_15.vd -aqmQStartFrom qbot_ep_15.vd -aqmAStartFrom abot_ep_15.vd
```

(Result.json generated)

```
python -m http.server 8000
```
(Visualize .json file in the browser)


## Modification based on reference code 

Combined with SL and RL modules, the `$Q$`-sampler samples the candidate question set from the output distribution of the RNN-based question-generator, `$p(q_t|h_{t-1})$`, using a beam search during every round. In order to approximate the information gain of each question, `$A_t$` is sampled from the approximated answer-generator network as several generated answers, while `$C_t$` is sampled from class posterior as several posterior test images.
We also apply an encoder module to understand the facts of input image features and related captions as well as past dialogues and convert them into hidden states and output vectors. Then we refer to the concept of attention model and give weights to the outputs of LSTM module in the decoder in order to focus more on the useful information.  In this way, we are able to generate the conditional probability distribution of the candidate questioner, which consequently gives the most likely original questions.

## Acknowlegement

This code is based on the github repository

_[N. Modhe, V. Prrabhu, M. Cogswell, S. Kottur, A. Das, S. Lee, D. Parikh, and D. Batra, VisDial-RL-PyTorch, 2018](https://github.com/batra-mlp-lab/visdial-rl.git)_

which is the official PyTorch implementation of

_[A. Das, S. Kottur, J. Moura, S. Lee, and D. Batra, Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning, 2017](https://arxiv.org/abs/1703.06585)_.

We would like to thank the authors of the work.

Copyright (c) NAVER Corp.
Licensed under [BSD 3-clause](LICENSE.md)

