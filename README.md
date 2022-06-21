# Risk Stratification and Pathway Analysis based on Interpretable Graph Representation Learning

Bilin Liang1, Haifan Gong1,2, Lanxuan Liu1, Lu Lu1, Jie Xu1, *

1 Shanghai Artificial Intelligence Laboratory, Shanghai, China

2 Sun Yat-sen University, School of Computer Science and Engineering, China

\* To whom correspondence should be addressed.

## Introduction of PathGNN

Pathway-based analysis of transcriptomic data has shown greater stability of biological activities and better performance than traditional gene-based analysis. Though a number of pathway-based deep learning models have been developed for bioinformatic analysis, topological information in pathways is still inaccessible, which limits the performance of the final prediction result, particularly in predicting disease outcomes using these models. To address this issue, we propose a novel model, called PathGNN, which constructs an interpretable Graph Representation Learning (GRL) model that can capture topological information hidden in pathway. PathGNN showed promising predictive performance in differentiating between long-term survival (LTS) and non-LTS when applied four types of cancer. The adoption of an interpretation algorithm enabled the identification of plausible pathways associated with survival. In summary, PathGNN demonstrates that GRL can be effectively applied to build a pathway-based model, resulting in promising predictive power. 

<div align=center>
<img src="https://github.com/BioAI-kits/PathGNN/blob/main/Figure/Figure.png" />
</div>

## Dependence

To use PathGNN, some dependences should be installed firstly, which includes Python (version, 3.9); Pytorch (version, 1.8); Pytorch Geometric (version, 2.0.3); captum; pandas; numpy; mygene; lifelines; sklearn. 

Besides, R and two library (GSVA, limma) for R should be installed. 

## To perform PathGNN

**Parameters**

Parameters are configured via a file suffixed with `.ini`, like LUAD.ini file.

**Building pathway graphs**

This step is to build pathway graphs which are the input of PathGNN. (Due to dataset size, we splited LUAD dataset into three parts. Thus, merging them to obtain complete dataset. More details information refer: https://github.com/BioAI-kits/PathGNN/blob/main/Data/LUAD/clean/readme.md)

```py
python data.py LUAD.ini
```

## 


The gene expression and clinical datasets download from TCGA (https://portal.gdc.cancer.gov/); The pathway information download from Reactome database (https://reactome.org/). Then prepare using `data.py`. 

The preprocessed dataset is very large, you can contact us if you need it (liangbilin@pjlab.org.cn or xujie@pjlab.org.cn).

we applied `model.py` to construct PathGNN models for predicting risk stratification of cancer. `.ini` files were profiles for building model.

`submodel.py` is used to realize the `Subnetwork1` in our paper.




All rights reserved.

