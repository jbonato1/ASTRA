
## Astro3S step by step notebooks
This software is composed by 2 pipelines named training and inference.

<div style="text-align:center"><img src="../github_images/test_train.svg" width="700" alt="Pipelines"/>

Astro3S hyperparameters are optimized on the training set pre-processed. Set_parameters, Training_Pipeline_PP and Training_PIpeline_Training_DNN are the notebooks where Training procedure is described. 


- Set_parameters: in this notebook it is described how to set all the hyperparameters of Astro3S using the training set 
- Training_Pipeline_PP: in this notebook it is described how to perform the preprocessing of training pipeline in Astro3S
- Training_Pipeline_Training_DNN: in this notebook it is described how to train the DNN


The inference pipeline comprises three main blocks - pre-processing, semantic segmentation, 
and subcellular cross-correlation analysis - allowing an unbiased end-to-end characterization of the complex morphological and dynamical properties of astrocytes.  


- Inference_Pipeline: In this notebook it is described how to perform AstroSS inference on the inference dataset
- CC_Pipeline: perform sub-cellular cross-correlation analysis on detected astrocytes

### Pre-processing modules results

<img src="../github_images/D1_pp_st.png" alt="Pre-proc"/>

### Segmentation Results
| Semantic Segmentation      | 
|:------------:|
|  <img src="../github_images/D1_sampleA.png" width="600"> |

|Single cell Details|
|:------------------:|
|  <img src="../github_images/D1_res_ex.png" width="600"> |

### Cross Correlation analysis Results
<img src="../github_images/D1_cc_.png" width="600" alt="Cross_corr"/>

