## Advanced GAN Project (WGAN with Gradient Penalty)


### TODO LOOM VIDEO



Repo Overview
- images/ - contains image results for each experiment
- models/ - contains model architecture for each model
- notebooks/ - contains notebooks for each model/experiment
  - data/ - contains downloaded CelebADataset
  - logs/ - contains Tensorboard logs (at least if I ran them locally, if not this is where they'll be stored when the notebooks are run)
- observations/ - contains observations/Takeaways for each model and transfer_of_knowledge.txt
- README.md - contains overview of repo as well as justification of choice of dataset

### Why I chose CelebADataset?
At first, I tried to make an SRGAN on the Div2K dataset. However, after creating it, I realized that the computational power required would be immense and would, frankly, take too much time given the contraints of the project. As such I chose a simpler architecture that added only a few steps from the initial DCGAN model, the WGAN model with gradient penalty variant, resulting in a much more simple but more attainable project

