# One Shot Learning

One Shot Learning using Omniglot data set and Siamese Network

This repository contains an implementation version of the paper [koch et al, Siamese Networks for one-shot learning](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

The model has been created using [keras](https://keras.io/).

The codebase has been divided in the following components:

- OmniglotService: Download the Omniglot data from the repository.
- OmniglotLoader: Load the Omniglot data based on categories.
- OmniglotDataset: Expose an api to process Omniglot data set.
- SiameseNet: Create a SiameseNetwork with the default configuration.
- OneShotRunner: Run One-shot learning experiments using Omniglot data set and siamese network.
- RunnerVisualizer: To display runner results.
- Utils: I/O funtionality to read/save state/data.

Details about the machine learning problem and the implementation is explained in more details in the [experiment](blank) notebook.

## Disclaimer

This is an early version of a work in progress prototype.

## Roadmap

- Allow Data configuration (normalization, resizing and augmentation...)
- Allow more configuration parameters for the runner
- Implement a NetworkConfigurator component for hyperparameter tunning
- Implement another model using a different approach