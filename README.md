# DeepMTMV
* Deep Multimodality Model for Multi-task Multi-view Learning
* https://arxiv.org/abs/1901.08723

# Summary:
* This project develops Deep-MTMV algorithm dealing with the multi-view multi-task classification problem.

# Dataset
* Webkb dataset can be downloaded from the following link:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/
* Please extract webkb dataset in ./DeepMTMV-master/data/

# To run the code:
* The code is written under python 3.6 with Pytorch framework and it could be ran by executing the following command:
* python DeepMTMV.py


# Important Flags in the python scripts:
* -g: Specify the index of GPU to use if there are multiple GPUs. The default is 0;
* -d: Specify the directory of the Webkb dataset;
* -s: Specify the number of sample used during the training;

# Note:
* The code requires the execution of Word2vec embedding, so Linux system or Mac OS is recommended.
