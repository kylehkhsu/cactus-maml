# CACTUs-MAML
CACTUs-MAML: Clustering to Automatically Generate Tasks for Unsupervised Model-Agnostic Meta-Learning.

This code was used to produce the CACTUs-MAML results and baselines in the paper [Unsupervised Learning via Meta-Learning](https://arxiv.org/abs/1810.02334).

This repository was built off of [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://github.com/cbfinn/maml).

### Dependencies
The code was tested with the following setup:
* Ubuntu 16.04
* Python 3.5.2
* Tensorflow-GPU 1.10

You can set up a Python virtualenv, activate it, and install the dependencies like so:
```
virtualenv venv --python=/usr/bin/python3
source venv/bin/activate
pip install -r requirements.txt
```

### Data
The Omniglot splits with ACAI and BiGAN encodings, MNIST splits with ACAI encodings, and miniImageNet splits with DeepCluster encodings used for the results in the paper are available [here](https://drive.google.com/open?id=1SbJQQ56FqfJVgy2DMynR60IH_bQHjW5m).
Download and extract the archive's contents into this directory.

Unfortunately, due to licensing issues, I am not at liberty to re-distribute the miniImageNet or CelebA datasets. The code for these datasets is still presented for posterity.

### Usage
You can find examples of scripts in ```/scripts```. Metrics can be visualized using Tensorboard. Evaluation results are saved to a .csv file in a run's log folder. All results were obtained using a single GPU.

### Credits
The unsupervised representations were computed using four open-source codebases from prior works.
* [Adversarial Feature Learning](https://github.com/jeffdonahue/bigan)
* [Deep Clustering for Unsupervised Learning of Visual Features](https://github.com/facebookresearch/deepcluster)
* [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://github.com/openai/InfoGAN)
* [Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](https://github.com/brain-research/acai)


### Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/hsukyle/cactus-maml/issues).

