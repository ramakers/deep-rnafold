*****************
# README
*****************

You will need the following tf versions:

tensorflow==1.14.0
tensorflow-gpu==1.14.0
tensorflow-probability==0.7.0
dm-sonnet==1.36

************************
## Data
************************

The raw data for training the embeddings is placed in ./data/substructures_drifts.pkl
and consists of three pandas DataFrames for train, validation and test set.

************************
## VQVAE
************************
training files and checkpoints for the vqvae are placed in ./vqvae

Once trained you can use the VQVAE to generate embeddings. With the ckpt provided
./data/npy_vqvae_embeddings_28022023.pkl has been generated which consist of
embeddings, masks, sequences, shape and homolog information for every sample
and serves as the data for the generator.
The same train / val / test split as for the raw data has been sustained, the data is just
in compact .npy format for faster training


*************************
## generator
*************************

under ./generator/train_generator.py is the file for training the residual network under masked target objective

*************************
## inference / mcts
*************************
1. for puzzle submissions we execute ./mcts/mcts.py to sample target embeddings based solely on sequence input
rollout_refinements.py is then used to generate refined structures as pdb files using SimRNA
2. in clustering/clustering.py we use a standard SimRNA clustering scheme to filter out five candidate structures
from all rollouts and simrna replica runs 
3. the five candidates can be further ranked with the discriminator network. for every puzzle to participate in a blind 
submission, each group is allowed to submit five models (hence five clusters)