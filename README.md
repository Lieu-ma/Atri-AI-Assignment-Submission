# Atri-AI-Assignment-Submission by Malavika N
Link to report: https://api.wandb.ai/links/malavika-n/n074s2m9

# Fashion-MNIST Neural Network from Scratch

This repository contains a NumPy implementation of a feedforward neural network for Fashion-MNIST classification.

The assignment focuses on implementing the network, backpropagation, and optimization algorithms from scratch. Keras is used only to load the Fashion-MNIST and MNIST datasets.

## Repository Contents

```text
.
├── README.md
├── Atri_FashionMNIST_Assignment.ipynb
├── neural_network.py
├── train.py
├── sweep.yaml
├── requirements.txt
├── neural_network.cpython-313.pyc
└── train.cpython-313.pyc
```

## What is Implemented

- Fashion-MNIST loading and sample visualization
- Flexible feedforward neural network
- Forward propagation
- Backpropagation using NumPy
- Mini-batch training
- Cross-entropy loss
- Squared error loss comparison
- Optimizers: SGD, Momentum, Nesterov, RMSProp, Adam, Nadam
- W&B sweep for hyperparameter tuning
- Final test-set evaluation
- Confusion matrix
- MNIST transfer experiments with three selected configurations

## Dataset

The main dataset is Fashion-MNIST. Each image is a 28 x 28 grayscale image and is flattened into a 784-dimensional vector before being passed into the neural network.

For Question 10, MNIST is used to test whether the hyperparameter choices learned from Fashion-MNIST transfer to another dataset.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Open the notebook in Google Colab:

```text
Atri_FashionMNIST_Assignment.ipynb
```

Run the cells in order. The notebook is organized question-wise from Question 1 to Question 10.

## Running the Training Script

To train the default best configuration:

```bash
python train.py
```

To run a specific configuration:

```bash
python train.py \
  --epochs 10 \
  --num_hidden_layers 4 \
  --hidden_size 128 \
  --activation tanh \
  --optimizer rmsprop \
  --learning_rate 0.001 \
  --batch_size 64 \
  --weight_init xavier \
  --weight_decay 0.0005
```

To run without W&B logging:

```bash
python train.py --no_wandb
```

## Running W&B Sweeps

Login to W&B first:

```bash
wandb login
```

Create a sweep:

```bash
wandb sweep sweep.yaml
```

Then run the agent using the sweep ID printed by W&B:

```bash
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID>
```


