# ODA Federated Learning

## Introduction

This algorithm optimizes the parameterization of Bayesian Networks generated by Pomegranate by sequentially updating the hyperparameters in different permutations iteratively, rewarding or penalizing hyperparameter probabilities based on model performance. Each iteration, the model is evaluated using ROC-AUC scoring provided a partition of training and validation sets generated at the start of each iteration.  The algorithm is intended to run in parallel with n number of processes and sample parameters from other processes using an asynchronous Gibbs sampling algorithm. 

## Technologies 

The federation algorithm is built-on the Bayesian Networks generated by <a href="https://pomegranate.readthedocs.io/en/latest/">Pomegranate.</a>

## How to Use

Download <code>ODA_federation_network.py</code>

Run <pre><code>process.py</code></pre> as Bash script followed by filename of dataset

### Example
In command line:
<pre><code>python process.py example_dataset.csv</code></pre>
