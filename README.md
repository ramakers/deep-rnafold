# Code & Data repository

### De Novo Prediction of RNA 3D Structures with Deep Generative Models


##### Usage:

# resnet.ipynb
This is a notebook that produces the embedding predictions based on sequence alone (+ homologs + SHAPE optionally).
As showcases we have plotted all results for the RNApuzzles.

# discr.ipynb
This notebook ranks generator predictions genererated e.g. by mcts based sampling.

# vqvae.ipynb
This notebook decodes the embeddings into pairwise distance matrices. We have also supplied code that relax and transform thos matrices into real space coordinates. Calculations for the RNApuzzles in the test set are shown.
