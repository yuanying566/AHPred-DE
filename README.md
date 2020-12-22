Code for the paper "A Deep Ensemble Predictor for Identifying Anti-Hypertensive Peptides Using Pretrained Protein Embedding"

1. Overview 

This repository is containing a main implementation of our model.

2. Dataset

Our dataset takes the orgainized dataset from Manavalan et al.[1]. 

3. The Pretrained Protein Model 

The pretrained protein model we used in our model is UniRep[2], which is based on 1900 layers of long-/-short-term-memory (LSTM) recurrent neural network (RNN) to try to predict the next amino acid through the previous amino acids predicted. In our model, we take the 64-unit UniRep model. For further information, please refer to https://github.com/churchlab/UniRep.

4. Comparison Models

We compared our model with three existing models mAHTPred[1], and AHTpin_AAC and AHTpin_ATC included in AHTpin[3]. The rest two models are mainly based on the different feature extraction methods. 




[1]	B. Manavalan, S. Basith, T. H. Shin, L. Wei, and G. Lee, “mAHTPred: a sequence-based meta-predictor for improving the prediction of anti-hypertensive peptides using effective feature representation,” Bioin-formatics, Dec 24, 2018.

[2] E. C. Alley, G. Khimulya, S. Biswas, M. AlQuraishi, and G. M. Church, “Unified rational protein engineering with sequence-based deep rep-resentation learning,” Nature Methods, vol. 16, no. 12, pp. 1315-+, Dec, 2019.

[3] R. Kumar, K. Chaudhary, J. Singh Chauhan, G. Nagpal, R. Kumar, M. Sharma, and G. P. Raghava, “An in silico platform for predicting, screening and designing of antihypertensive peptides,” Sci Rep, vol. 5, pp. 12512, Jul 27, 2015.




