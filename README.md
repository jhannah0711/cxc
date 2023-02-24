# Chicken Nuggets 🐔🐔🐔🐔
This is our CxC Hackathon submission for the Cyclica challenge.

The most important features for drug-binding are information on the interaction between the protein and ligand. Therefore, columns named feat_PHI, feat_PSI, feat_TAU, feat_THETA which indicate protein chain bonding angles, and feat_BBSASA, feat_SCSASA which shows the solvent accessible surface area should be combined to train the model. The way we handle the data is that we replace the zero value with its median to increase the accuracy as well as the f1 score. 
