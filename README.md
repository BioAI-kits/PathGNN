# PathGNN

PathGNN： Risk Stratification and Pathway Analysis based on Interpretable Graph Representation Learning

The gene expression and clinical datasets download from TCGA (https://portal.gdc.cancer.gov/); The pathway information download from Reactome database (https://reactome.org/). Then prepare using `data.py`. The preprocessed dataset is very large, you can contact us if you need it (liangbilin@pjlab.org.cn).

we applied `model.py` to construct PathGNN models for predicting risk stratification of cancer. `.ini` files were profiles for building model.

`submodel.py` is used to realize the `Subnetwork1` in our paper.

All rights reserved.
