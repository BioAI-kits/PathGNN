# Risk Stratification and Pathway Analysis based on Interpretable Graph Representation Learning

Bilin Liang1, Haifan Gong1,2, Lanxuan Liu1, Lu Lu1, Jie Xu1, *

1 Shanghai Artificial Intelligence Laboratory, Shanghai, China

2 Sun Yat-sen University, School of Computer Science and Engineering, China

\* To whom correspondence should be addressed.


The gene expression and clinical datasets download from TCGA (https://portal.gdc.cancer.gov/); The pathway information download from Reactome database (https://reactome.org/). Then prepare using `data.py`. The preprocessed dataset is very large, you can contact us if you need it (liangbilin@pjlab.org.cn or xujie@pjlab.org.cn).

we applied `model.py` to construct PathGNN models for predicting risk stratification of cancer. `.ini` files were profiles for building model.

`submodel.py` is used to realize the `Subnetwork1` in our paper.


![](https://github.com/BioAI-kits/PathGNN/blob/main/fig1.png)

All rights reserved.

