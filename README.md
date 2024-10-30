
# Atlas-GAN


Atlas-GAN is a cutting-edge generative adversarial network (GAN) project designed to explore image generation. It leverages deep learning to produce highly detailed and realistic outputs.

### About the Developer
Hi, I'm David Meddaugh - a machine learning student and developer with a background in education


### Project Description
This GAN architecture utilizes a two-model adversarial setup to iteratively improve output quality through refined generator and discriminator cycles. The project focuses on improving output realism, handling high-resolution data, and exploring unsupervised learning methodologies.

The following repository contains two projects

    1. A Basic Deep Convolutional GAN architecture that focuses on recreating images from the MNIST dataset

    2. A more advanced WGAN with Gradient Penalty variant that focuses on creating images from the CelebADataset

The project focused on created a working model and then performing experiments, after which I wrote some basic observations on the effects of the modified architecture or hyperparameters. 


### Challenges Faced
One of the most challenging aspects of Atlas-GAN has been achieving balance between the generator and discriminator networks. Because they compete against each other in an unsupervised training environment, this is too be expected. This was most apparent in the WGAN project (Exp #1 I believe) in which the discriminator loss steadily decreased while the generator loss oscillated wildly. 



Getting Started

Clone the repository:


```git clone https://github.com/meddizzle316/atlas-gan.git``` 

```cd atlas-gan```

You'll need to make sure you have pyTorch, Cuda, a jupyter Notebook extension (this is free in VS Code but I used Pycharm Pro) and a local GPU if you want a reasonable training time (though it will still work on a CPU, just much slower). The training sets are downloaded automatically as part of the script


Run the model:

Each of the 'notebooks' included in the notebook directory are self contained and can be run so long as you have the requirements installed
