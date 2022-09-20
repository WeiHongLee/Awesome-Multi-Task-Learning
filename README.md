# Awesome Multi-task Learning

Feel free to contact me or contribute if you find any interesting paper is missing!

## Table of Contents
- [Survey & Study](#survey--study)
- [Benchmarks & Code](#benchmarks--code)
- [Papers](#papers)
- [Awesome Multi-domain Multi-task Learning](#awesome-multi-domain-multi-task-learning)
- [Workshops](#workshops)
- [Online Courses](#online-courses)
- [Related awesome list](#related-awesome-list)

## Survey & Study

* Factors of Influence for Transfer Learning across Diverse Appearance Domains and Task Types (TPAMI, 2022) [[paper](https://arxiv.org/pdf/2103.13318.pdf)]

* Multi-Task Learning for Dense Prediction Tasks: A Survey (TPAMI, 2021) [[paper](https://arxiv.org/abs/2004.13379)] [[code](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]

* A Survey on Multi-Task Learning (TKDE, 2021) [[paper](https://ieeexplore.ieee.org/abstract/document/9392366)]

* Multi-Task Learning with Deep Neural Networks: A Survey (arXiv, 2020) [[paper](http://arxiv.org/abs/2009.09796)]

* Taskonomy: Disentangling Task Transfer Learning (CVPR, 2018 [best paper]) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.pdf)] [[dataset](http://taskonomy.stanford.edu/)]

* A Comparison of Loss Weighting Strategies for Multi task Learning in Deep Neural Networks (IEEE Access, 2019) [[paper](https://ieeexplore.ieee.org/document/8848395)]

* An Overview of Multi-Task Learning in Deep Neural Networks (arXiv, 2017) [[paper](http://arxiv.org/abs/1706.05098)]

## Benchmarks & Code
<details>
  <summary>Benchmarks</summary>

### Dense Prediction Tasks

* **[NYUv2]** Indoor Segmentation and Support Inference from RGBD Images (ECCV, 2012) [[paper](https://cs.nyu.edu/~silberman/papers/indoor_seg_support.pdf)] [[dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)]

* **[Cityscapes]** The Cityscapes Dataset for Semantic Urban Scene Understanding (CVPR, 2016) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7780719)] [[dataset](https://www.cityscapes-dataset.com/)]

* **[PASCAL-Context]** The Role of Context for Object Detection and Semantic Segmentation in the Wild (CVPR, 2014) [[paper](https://cs.stanford.edu/~roozbeh/pascal-context/mottaghi_et_al_cvpr14.pdf)] [[dataset](https://cs.stanford.edu/~roozbeh/pascal-context/)]

* **[Taskonomy]** Taskonomy: Disentangling Task Transfer Learning (CVPR, 2018 [best paper]) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zamir_Taskonomy_Disentangling_Task_CVPR_2018_paper.pdf)] [[dataset](http://taskonomy.stanford.edu/)]

* **[KITTI]** Vision meets robotics: The KITTI dataset (IJRR, 2013) [[paper](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)] [dataset](http://www.cvlibs.net/datasets/kitti/)

* **[SUN RGB-D]** SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite (CVPR 2015) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298655)] [[dataset](https://rgbd.cs.princeton.edu)]

* **[BDD100K]** BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning (CVPR, 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.pdf)] [[dataset](https://bdd-data.berkeley.edu/)]

* **[Omnidata]** Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans (ICCV, 2021) [[paper](https://arxiv.org/pdf/2110.04994.pdf)] [[project](https://omnidata.vision)]

### Image Classification

* **[Meta-dataset]** Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples (ICLR, 2020) [[paper](https://openreview.net/pdf?id=rkgAGAVKPr)] [[dataset](https://github.com/google-research/meta-dataset)]

* **[Visual Domain Decathlon]** Learning multiple visual domains with residual adapters (NeurIPS, 2017) [[paper](https://arxiv.org/abs/1705.08045)] [[dataset](https://www.robots.ox.ac.uk/~vgg/decathlon/)]

* **[CelebA]** Deep Learning Face Attributes in the Wild (ICCV, 2015) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7410782)] [[dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)]
</details>


<details>
  <summary>Code</summary>

* [[Multi-Task-Learning-PyTorch](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]: Multi-task Dense Prediction.

* [[Auto-λ](https://github.com/lorenmt/auto-lambda)]: Multi-task Dense Prediction, Robotics.

* [[UniversalRepresentations](https://github.com/VICO-UoE/UniversalRepresentations)]: [Multi-task Dense Prediction](https://github.com/VICO-UoE/UniversalRepresentations/tree/main/DensePred) (including different loss weighting strategies), [Multi-domain Classification](https://github.com/VICO-UoE/UniversalRepresentations/tree/main/VisualDecathlon), [Cross-domain Few-shot Learning](https://github.com/VICO-UoE/URL).

* [[MTAN](https://github.com/lorenmt/mtan)]: Multi-task Dense Prediction, Multi-domain Classification.

* [[ASTMT](https://github.com/facebookresearch/astmt)]: Multi-task Dense Prediction.

* [[LibMTL](https://github.com/median-research-group/libmtl)]: Multi-task Dense Prediction.

* [[MTPSL](https://github.com/VICO-UoE/MTPSL)]: Multi-task Partially-supervised Learning for Dense Prediction.

* [[Resisual Adapater](https://github.com/srebuffi/residual_adapters)]: Multi-domain Classification.
</details>

## Papers

### 2022

* Task Discovery: Finding the Tasks that Neural Networks Generalize on (NeurIPS, 2022) 

* **[Auto-λ]** Auto-λ: Disentangling Dynamic Task Relationships (TMLR, 2022) [[paper](https://arxiv.org/pdf/2202.03091.pdf)] [[code](https://github.com/lorenmt/auto-lambda)]

* **[Universal Representations]** Universal Representations: A Unified Look at Multiple Task and Domain Learning (arXiv, 2022) [[paper](https://arxiv.org/pdf/2204.02744.pdf)] [[code](https://github.com/VICO-UoE/UniversalRepresentations)]

* Not All Models Are Equal: Predicting Model Transferability in a Self-challenging Fisher Space (ECCV, 2022) [[paper](https://arxiv.org/abs/2207.03036)] [[code](https://github.com/TencentARC/SFDA)]

* Factorizing Knowledge in Neural Networks (ECCV, 2022) [[paper](https://arxiv.org/abs/2207.03337)] [[code](https://github.com/Adamdad/KnowledgeFactor)]

* **[InvPT]** Inverted Pyramid Multi-task Transformer for Dense Scene Understanding (ECCV, 2022) [[paper](https://arxiv.org/pdf/2203.07997.pdf)] [[code](https://github.com/prismformore/InvPT)]

* **[MultiMAE]** MultiMAE: Multi-modal Multi-task Masked Autoencoders (ECCV, 2022) [[paper](https://arxiv.org/pdf/2204.01678.pdf)] [[code](https://multimae.epfl.ch)]

* A Multi-objective / Multi-task Learning Framework Induced by Pareto Stationarity (ICML, 2022) [[paper](https://proceedings.mlr.press/v162/momma22a.html)]

* Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization (ICML, 2022) [[paper](https://proceedings.mlr.press/v162/javaloy22a.html)]

* Active Multi-Task Representation Learning (ICML, 2022) [[paper](https://proceedings.mlr.press/v162/chen22j.html)]

* Generative Modeling for Multi-task Visual Learning (ICML, 2022) [[paper](https://proceedings.mlr.press/v162/bao22c.html)] [[code](https://github.com/zpbao/multi-task-oriented_generative_modeling)]

* Multi-Task Learning as a Bargaining Game (ICML, 2022) [[paper](https://proceedings.mlr.press/v162/navon22a.html)] [[code](https://github.com/AvivNavon/nash-mtl)]

* Multi-Task Learning with Multi-query Transformer for Dense Prediction (arXiv, 2022) [[paper](https://arxiv.org/pdf/2205.14354.pdf)]

* **[Gato]** A Generalist Agent (arXiv, 2022) [[paper](https://arxiv.org/pdf/2205.06175.pdf)]

* **[MTPSL]** Learning Multiple Dense Prediction Tasks from Partially Annotated Data (CVPR, 2022) [[paper](https://arxiv.org/pdf/2111.14893.pdf)] [[code](https://github.com/VICO-UoE/MTPSL)]

* **[TSA]** Cross-domain Few-shot Learning with Task-specific Adapters (CVPR, 2022) [[paper](https://arxiv.org/pdf/2107.00358.pdf)] [[code](https://github.com/VICO-UoE/URL)]

* **[OMNIVORE]** OMNIVORE: A Single Model for Many Visual Modalities (CVPR, 2022) [[paper](https://arxiv.org/pdf/2201.08377.pdf)] [[code](https://github.com/facebookresearch/omnivore)]

* Task Adaptive Parameter Sharing for Multi-Task Learning (CVPR, 2022) [[paper](https://arxiv.org/pdf/2203.16708.pdf)]

* Controllable Dynamic Multi-Task Architectures (CVPR, 2022) [[paper](https://arxiv.org/pdf/2203.14949.pdf)] [[code](https://www.nec-labs.com/~mas/DYMU/)]

* **[SHIFT]** SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation (CVPR, 2022) [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_SHIFT_A_Synthetic_Driving_Dataset_for_Continuous_Multi-Task_Domain_Adaptation_CVPR_2022_paper.pdf)] [[code](https://www.vis.xyz/shift/)]

* DiSparse: Disentangled Sparsification for Multitask Model Compression (CVPR, 2022) [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_DiSparse_Disentangled_Sparsification_for_Multitask_Model_Compression_CVPR_2022_paper.pdf)] [[code](https://github.com/SHI-Labs/DiSparse-Multitask-Model-Compression)]

* **[MulT]** MulT: An End-to-End Multitask Learning Transformer (CVPR, 2022) [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Bhattacharjee_MulT_An_End-to-End_Multitask_Learning_Transformer_CVPR_2022_paper.pdf)] [[code](https://github.com/IVRL/MulT)]

* Sound and Visual Representation Learning with Multiple Pretraining Tasks (CVPR, 2022) [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Vasudevan_Sound_and_Visual_Representation_Learning_With_Multiple_Pretraining_Tasks_CVPR_2022_paper.pdf)]

* Medusa: Universal Feature Learning via Attentional Multitasking (CVPR Workshop, 2022) [[paper](https://arxiv.org/abs/2204.05698)]

* An Evolutionary Approach to Dynamic Introduction of Tasks in Large-scale Multitask Learning Systems (arXiv, 2022) [[paper](https://arxiv.org/pdf/2205.12755.pdf)] [[code](https://github.com/google-research/google-research/tree/master/muNet)]

* Combining Modular Skills in Multitask Learning (arXiv, 2022) [[paper](https://arxiv.org/pdf/2202.13914.pdf)]

* Visual Representation Learning over Latent Domains (ICLR, 2022) [[paper](https://openreview.net/pdf?id=kG0AtPi6JI1)]

* ADARL: What, Where, and How to Adapt in Transfer Reinforcement Learning (ICLR, 2022) [[paper](https://openreview.net/pdf?id=8H5bpVwvt5)] [[code](https://github.com/Adaptive-RL/AdaRL-code)]

* Towards a Unified View of Parameter-Efficient Transfer Learning (ICLR, 2022) [[paper](https://openreview.net/pdf?id=0RDcd5Axok)] [[code](https://github.com/jxhe/unify-parameter-efficient-tuning)]

* **[Rotograd]** Rotograd: Dynamic Gradient Homogenization for Multi-Task Learning (ICLR, 2022) [[paper](https://openreview.net/pdf?id=T8wHz4rnuGL)] [[code](https://github.com/adrianjav/rotograd)]

* Relational Multi-task Learning: Modeling Relations Between Data and Tasks (ICLR, 2022) [[paper](https://openreview.net/pdf?id=8Py-W8lSUgy)]

* Weighted Training for Cross-task Learning (ICLR, 2022) [[paper](https://openreview.net/pdf?id=ltM1RMZntpu)] [[code](https://github.com/CogComp/TAWT)]

* Semi-supervised Multi-task Learning for Semantics and Depth (WACV, 2022) [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Wang_Semi-Supervised_Multi-Task_Learning_for_Semantics_and_Depth_WACV_2022_paper.pdf)]

* In Defense of the Unitary Scalarization for Deep Multi-Task Learning (arXiv, 2022) [[paper](https://arxiv.org/pdf/2201.04122.pdf)]

### 2021

* Efficiently Identifying Task Groupings for Multi-Task Learning (NeurIPS, 2021) [[paper](http://arxiv.org/abs/2109.04617)]

* **[CAGrad]** Conflict-Averse Gradient Descent for Multi-task Learning (NeurIPS, 2021) [[paper](https://openreview.net/pdf?id=_61Qh8tULj_)] [[code](https://github.com/Cranial-XIX/CAGrad)]

* A Closer Look at Loss Weighting in Multi-Task Learning (arXiv, 2021) [[paper](https://arxiv.org/pdf/2111.10603.pdf)]

* Exploring Relational Context for Multi-Task Dense Prediction (ICCV, 2021) [[paper](http://arxiv.org/abs/2104.13874)] [[code](https://github.com/brdav/atrc)]

* Multi-Task Self-Training for Learning General Representations (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ghiasi_Multi-Task_Self-Training_for_Learning_General_Representations_ICCV_2021_paper.pdf)]

* Task Switching Network for Multi-task Learning (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Task_Switching_Network_for_Multi-Task_Learning_ICCV_2021_paper.html)] [[code](https://github.com/GuoleiSun/TSNs)]

* Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans (ICCV, 2021) [[paper](https://arxiv.org/pdf/2110.04994.pdf)] [[project](https://omnidata.vision)]

* Robustness via Cross-Domain Ensembles (ICCV, 2021) [[paper](https://arxiv.org/abs/2103.10919)] [[code](https://github.com/EPFL-VILAB/XDEnsembles)]

* Domain Adaptive Semantic Segmentation with Self-Supervised Depth Estimation (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Domain_Adaptive_Semantic_Segmentation_With_Self-Supervised_Depth_Estimation_ICCV_2021_paper.pdf)] [[code](https://qin.ee/corda)]

* **[URL]** Universal Representation Learning from Multiple Domains for Few-shot Classification (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Universal_Representation_Learning_From_Multiple_Domains_for_Few-Shot_Classification_ICCV_2021_paper.pdf)] [[code](https://github.com/VICO-UoE/URL)]

* **[tri-M]** A Multi-Mode Modulator for Multi-Domain Few-Shot Classification (ICCV, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_A_Multi-Mode_Modulator_for_Multi-Domain_Few-Shot_Classification_ICCV_2021_paper.pdf)] [[code](https://github.com/csyanbin/tri-M-ICCV)]

* MultiTask-CenterNet (MCN): Efficient and Diverse Multitask Learning using an Anchor Free Approach (ICCV Workshop, 2021) [[paper](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/papers/Heuer_MultiTask-CenterNet_MCN_Efficient_and_Diverse_Multitask_Learning_Using_an_Anchor_ICCVW_2021_paper.pdf)]

* See Yourself in Others: Attending Multiple Tasks for Own Failure Detection (arXiv, 2021) [[paper](https://arxiv.org/pdf/2110.02549.pdf)]

* A Multi-Task Cross-Task Learning Architecture for Ad-hoc Uncertainty Estimation in 3D Cardiac MRI Image Segmentation (CinC, 2021) [[paper](https://www.cinc.org/2021/Program/accepted/115_Preprint.pdf)] [[code](https://github.com/SMKamrulHasan/MTCTL)]

* Multi-Task Reinforcement Learning with Context-based Representations (ICML, 2021) [[paper](http://arxiv.org/abs/2102.06177)]

* **[FLUTE]** Learning a Universal Template for Few-shot Dataset Generalization (ICML, 2021) [[paper](https://arxiv.org/pdf/2105.07029.pdf)] [[code](https://github.com/google-research/meta-dataset)]

* Towards a Unified View of Parameter-Efficient Transfer Learning (arXiv, 2021) [[paper](http://arxiv.org/abs/2110.04366)]

* UniT: Multimodal Multitask Learning with a Unified Transformer (arXiv, 2021) [[paper](http://arxiv.org/abs/2102.10772)]

* Learning to Relate Depth and Semantics for Unsupervised Domain Adaptation (CVPR, 2021) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Saha_Learning_To_Relate_Depth_and_Semantics_for_Unsupervised_Domain_Adaptation_CVPR_2021_paper.pdf)] [[code](https://github.com/susaha/ctrl-uda)]

* CompositeTasking: Understanding Images by Spatial Composition of Tasks (CVPR, 2021) [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Popovic_CompositeTasking_Understanding_Images_by_Spatial_Composition_of_Tasks_CVPR_2021_paper.html)] [[code](https://github.com/nikola3794/composite-tasking)]

* Anomaly Detection in Video via Self-Supervised and Multi-Task Learning (CVPR, 2021) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Georgescu_Anomaly_Detection_in_Video_via_Self-Supervised_and_Multi-Task_Learning_CVPR_2021_paper.pdf)]

* Taskology: Utilizing Task Relations at Scale (CVPR, 2021) [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lu_Taskology_Utilizing_Task_Relations_at_Scale_CVPR_2021_paper.pdf)]

* Three Ways to Improve Semantic Segmentation with Self-Supervised Depth Estimation (CVPR, 2021) [[paper](https://arxiv.org/pdf/2012.10782.pdf)] [[code](https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth)]

* Improving Semi-Supervised and Domain-Adaptive Semantic Segmentation with Self-Supervised Depth Estimation (arXiv, 2021) [[paper](https://arxiv.org/pdf/2108.12545.pdf)] [[code](https://github.com/lhoyer/improving_segmentation_with_selfsupervised_depth)]

* Counter-Interference Adapter for Multilingual Machine Translation (Findings of EMNLP, 2021) [[paper](https://aclanthology.org/2021.findings-emnlp.240)]

* Conditionally Adaptive Multi-Task Learning: Improving Transfer Learning in NLP Using Fewer Parameters & Less Data (ICLR) [[paper](https://openreview.net/forum?id=de11dbHzAMF)] [[code](https://github.com/CAMTL/CA-MTL)]

* **[Gradient Vaccine]** Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (ICLR, 2021) [[paper](https://openreview.net/forum?id=F1vEjWK-lH_)] 

* **[IMTL]** Towards Impartial Multi-task Learning (ICLR, 2021) [[paper](https://openreview.net/forum?id=IMPnRXEWpvr)]

* Deciphering and Optimizing Multi-Task Learning: A Random Matrix Approach (ICLR, 2021) [[paper](https://openreview.net/forum?id=Cri3xz59ga)]

* **[URT]** A Universal Representation Transformer Layer for Few-Shot Image Classification (ICLR, 2021) [[paper](https://arxiv.org/pdf/2006.11702.pdf)] [[code](https://github.com/liulu112601/URT)]

* Flexible Multi-task Networks by Learning Parameter Allocation (ICLR Workshop, 2021) [[paper](http://arxiv.org/abs/1910.04915)]

* Multi-Loss Weighting with Coefficient of Variations (WACV, 2021) [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Groenendijk_Multi-Loss_Weighting_With_Coefficient_of_Variations_WACV_2021_paper.pdf)] [[code](https://github.com/rickgroen/cov-weighting)]

### 2020

* Multi-Task Reinforcement Learning with Soft Modularization (NeurIPS, 2020) [[paper](http://arxiv.org/abs/2003.13661)] [[code](https://github.com/RchalYang/Soft-Module)] 
* AdaShare: Learning What To Share For Efficient Deep Multi-Task Learning (NeurIPS, 2020) [[paper](http://arxiv.org/abs/1911.12423)] [[code](https://github.com/sunxm2357/AdaShare)]

* **[GradDrop]** Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout (NeurIPS, 2020) [[paper](https://proceedings.NeurIPS.cc//paper/2020/file/16002f7a455a94aa4e91cc34ebdb9f2d-Paper.pdf)] [[code](https://github.com/tensorflow/lingvo/blob/master/lingvo/core/graddrop.py)]

* **[PCGrad]** Gradient Surgery for Multi-Task Learning (NeurIPS, 2020) [[paper](http://arxiv.org/abs/2001.06782)] [[tensorflow](https://github.com/tianheyu927/PCGrad)] [[pytorch](https://github.com/WeiChengTseng/Pytorch-PCGrad)]

* On the Theory of Transfer Learning: The Importance of Task Diversity (NeurIPS, 2020) [[paper](https://proceedings.NeurIPS.cc//paper/2020/file/59587bffec1c7846f3e34230141556ae-Paper.pdf)]

* A Study of Residual Adapters for Multi-Domain Neural Machine Translation (WMT, 2020) [[paper](https://www.aclweb.org/anthology/2020.wmt-1.72/)]

* Multi-Task Adversarial Attack (arXiv, 2020) [[paper](http://arxiv.org/abs/2011.09824)]

* Automated Search for Resource-Efficient Branched Multi-Task Networks (BMVC, 2020) [[paper](http://arxiv.org/abs/2008.10292)] [[code](https://github.com/brdav/bmtas)]
* Branched Multi-Task Networks: Deciding What Layers To Share (BMVC, 2020) [[paper](http://arxiv.org/abs/1904.02920)]

* MTI-Net: Multi-Scale Task Interaction Networks for Multi-Task Learning (ECCV, 2020) [[paper](http://arxiv.org/abs/2001.06902)] [[code](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch)]

* Reparameterizing Convolutions for Incremental Multi-Task Learning without Task Interference (ECCV, 2020) [[paper](http://arxiv.org/abs/2007.12540)] [[code](https://github.com/menelaoskanakis/RCM)]

* Selecting Relevant Features from a Multi-domain Representation for Few-shot Classification (ECCV, 2020) [[paper](https://arxiv.org/pdf/2003.09338.pdf)] [[code](https://github.com/dvornikita/SUR)]

* Multitask Learning Strengthens Adversarial Robustness (ECCV 2020) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470154.pdf)] [[code](https://github.com/columbia/MTRobust)]

* Duality Diagram Similarity: a generic framework for initialization selection in task transfer learning (ECCV, 2020) [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710494.pdf)] [[code](https://github.com/cvai-repo/duality-diagram-similarity)]

* **[KD4MTL]** Knowledge Distillation for Multi-task Learning (ECCV Workshop) [[paper](https://arxiv.org/pdf/2007.06889.pdf)] [[code](https://github.com/VICO-UoE/KD4MTL)]

* MTL-NAS: Task-Agnostic Neural Architecture Search towards General-Purpose Multi-Task Learning (CVPR, 2020) [[paper](https://arxiv.org/abs/2003.14058)] [[code](https://github.com/bhpfelix/MTLNAS)]

* Robust Learning Through Cross-Task Consistency (CVPR, 2020) [[paper](https://consistency.epfl.ch/Cross_Task_Consistency_CVPR2020.pdf)] [[code](https://github.com/EPFL-VILAB/XTConsistency)]

* 12-in-1: Multi-Task Vision and Language Representation Learning (CVPR, 2020) [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.pdf) [[code](https://github.com/facebookresearch/vilbert-multi-task)]

* A Multi-task Mean Teacher for Semi-supervised Shadow Detection (CVPR, 2020) [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Multi-Task_Mean_Teacher_for_Semi-Supervised_Shadow_Detection_CVPR_2020_paper.pdf)] [[code](https://github.com/eraserNut/MTMT)]

* MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer (EMNLP, 2020) [[paper](https://doi.org/10.18653/v1/2020.emnlp-main.617)]

* Masking as an Efficient Alternative to Finetuning for Pretrained Language Models (EMNLP, 2020) [[paper](http://arxiv.org/abs/2004.12406)] [[code](https://github.com/ptlmasking/maskbert)]

* Effcient Continuous Pareto Exploration in Multi-Task Learning (ICML, 2020) [[paper](http://proceedings.mlr.press/v119/ma20a/ma20a.pdf)] [[code](https://github.com/mit-gfx/ContinuousParetoMTL)]

* Which Tasks Should Be Learned Together in Multi-task Learning? (ICML, 2020) [[paper](http://arxiv.org/abs/1905.07553)] [[code](https://github.com/tstandley/taskgrouping)]

* Learning to Branch for Multi-Task Learning (ICML, 2020) [[paper](https://arxiv.org/abs/2006.01895)]

* Partly Supervised Multitask Learning (ICMLA, 2020) [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9356271)

* Understanding and Improving Information Transfer in Multi-Task Learning (ICLR, 2020) [[paper](https://arxiv.org/abs/2005.00944)]

* Measuring and Harnessing Transference in Multi-Task Learning (arXiv, 2020) [[paper](https://arxiv.org/abs/2010.15413)]

* Multi-Task Semi-Supervised Adversarial Autoencoding for Speech Emotion Recognition (arXiv, 2020) [[paper](https://arxiv.org/pdf/1907.06078.pdf)]

* Learning Sparse Sharing Architectures for Multiple Tasks (AAAI, 2020) [[paper](http://arxiv.org/abs/1911.05034)]

* AdapterFusion: Non-Destructive Task Composition for Transfer Learning (arXiv, 2020) [[paper](http://arxiv.org/abs/2005.00247)]

### 2019

* Adaptive Auxiliary Task Weighting for Reinforcement Learning (NeurIPS, 2019) [[paper](https://papers.nips.cc/paper/2019/hash/0e900ad84f63618452210ab8baae0218-Abstract.html)]

* Pareto Multi-Task Learning (NeurIPS, 2019) [[paper](http://papers.nips.cc/paper/9374-pareto-multi-task-learning.pdf)] [[code](https://github.com/Xi-L/ParetoMTL)]

* Modular Universal Reparameterization: Deep Multi-task Learning Across Diverse Domains (NeurIPS, 2019) [[paper](http://arxiv.org/abs/1906.00097)]

* Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes (NeurIPS, 2019) [[paper](https://github.com/cambridge-mlg/cnaps)] [[code](https://proceedings.neurips.cc/paper/2019/file/1138d90ef0a0848a542e57d1595f58ea-Paper.pdf)]

* **[Orthogonal]** Regularizing Deep Multi-Task Networks using Orthogonal Gradients (arXiv, 2019) [[paper](http://arxiv.org/abs/1912.06844)]

* Many Task Learning With Task Routing (ICCV, 2019) [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Strezoski_Many_Task_Learning_With_Task_Routing_ICCV_2019_paper.pdf)] [[code](https://github.com/gstrezoski/TaskRouting)]

* Stochastic Filter Groups for Multi-Task CNNs: Learning Specialist and Generalist Convolution Kernels (ICCV, 2019) [[paper](https://arxiv.org/abs/1908.09597)]

* Deep Elastic Networks with Model Selection for Multi-Task Learning (ICCV, 2019) [[paper](http://arxiv.org/abs/1909.04860)] [[code](https://github.com/rllab-snu/Deep-Elastic-Network)]

* Feature Partitioning for Efficient Multi-Task Architectures (arXiv, 2019) [[paper](https://arxiv.org/abs/1908.04339)] [[code](https://github.com/google/multi-task-architecture-search)]

* Task Selection Policies for Multitask Learning (arXiv, 2019) [[paper](http://arxiv.org/abs/1907.06214)]

* BAM! Born-Again Multi-Task Networks for Natural Language Understanding (ACL, 2019) [[paper](https://www.aclweb.org/anthology/P19-1595/)] [[code](https://github.com/google-research/google-research/tree/master/bam)]

* OmniNet: A unified architecture for multi-modal multi-task learning (arXiv, 2019) [[paper](http://arxiv.org/abs/1907.07804)]

* NDDR-CNN: Layerwise Feature Fusing in Multi-Task CNNs by Neural Discriminative Dimensionality Reduction (CVPR, 2019) [[paper](https://arxiv.org/abs/1801.08297)] [[code](https://github.com/ethanygao/NDDR-CNN)]

* **[MTAN + DWA]** End-to-End Multi-Task Learning with Attention (CVPR, 2019) [[paper](http://arxiv.org/abs/1803.10704)] [[code](https://github.com/lorenmt/mtan)] 

* Attentive Single-Tasking of Multiple Tasks (CVPR, 2019) [[paper](http://arxiv.org/abs/1904.08918)] [[code](https://github.com/facebookresearch/astmt)]

* Pattern-Affinitive Propagation Across Depth, Surface Normal and Semantic Segmentation (CVPR, 2019) [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Pattern-Affinitive_Propagation_Across_Depth_Surface_Normal_and_Semantic_Segmentation_CVPR_2019_paper.pdf)]

* Representation Similarity Analysis for Efficient Task Taxonomy & Transfer Learning (CVPR, 2019) [[paper](https://arxiv.org/abs/1904.11740)] [[code](https://github.com/kshitijd20/RSA-CVPR19-release)]

* **[Geometric Loss Strategy (GLS)]** MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR Workshop, 2019) [[paper](http://arxiv.org/abs/1904.08492)]

* Parameter-Efficient Transfer Learning for NLP (ICML, 2019) [[paper](http://arxiv.org/abs/1902.00751)]

* BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning (ICML, 2019) [[paper](http://arxiv.org/abs/1902.02671)] [[code](https://github.com/AsaCooperStickland/Bert-n-Pals)]

* Tasks Without Borders: A New Approach to Online Multi-Task Learning (ICML Workshop, 2019) [[paper](https://openreview.net/pdf?id=HkllV5Bs24)]

* AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning (NACCL, 2019) [[paper](https://arxiv.org/abs/1904.04153)] [[code](https://github.com/HanGuo97/AutoSeM)]

* Multi-Task Deep Reinforcement Learning with PopArt (AAAI, 2019) [[paper](https://doi.org/10.1609/aaai.v33i01.33013796)]

* SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning (AAAI, 2019) [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/3788/3666)]

* Latent Multi-task Architecture Learning (AAAI, 2019) [[paper](https://arxiv.org/abs/1705.08142)] [[code](https://github.com/ sebastianruder/sluice-networks)]

* Multi-Task Deep Neural Networks for Natural Language Understanding (ACL, 2019) [[paper](https://arxiv.org/pdf/1901.11504.pdf)]

### 2018

* Learning to Multitask (NeurIPS, 2018) [[paper](https://papers.nips.cc/paper/2018/file/aeefb050911334869a7a5d9e4d0e1689-Paper.pdf)]

* **[MGDA]** Multi-Task Learning as Multi-Objective Optimization (NeurIPS, 2018) [[paper](http://arxiv.org/abs/1810.04650)] [[code](https://github.com/isl-org/MultiObjectiveOptimization)]

* Adapting Auxiliary Losses Using Gradient Similarity (arXiv, 2018) [[paper](http://arxiv.org/abs/1812.02224)] [[code](https://github.com/szkocot/Adapting-Auxiliary-Losses-Using-Gradient-Similarity)]

* Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights (ECCV, 2018) [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Arun_Mallya_Piggyback_Adapting_a_ECCV_2018_paper.pdf)] [[code](https://github.com/arunmallya/piggyback)]

* Dynamic Task Prioritization for Multitask Learning (ECCV, 2018) [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Focus_on_the_ECCV_2018_paper.pdf)]

* A Modulation Module for Multi-task Learning with Applications in Image Retrieval (ECCV, 2018) [[paper](https://arxiv.org/abs/1807.06708)]

* Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD, 2018) [[paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)] 

* Unifying and Merging Well-trained Deep Neural Networks for Inference Stage (IJCAI, 2018) [[paper](http://arxiv.org/abs/1805.04980)] [[code](https://github.com/ivclab/NeuralMerger)]

* Efficient Parametrization of Multi-domain Deep Neural Networks (CVPR, 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Rebuffi_Efficient_Parametrization_of_CVPR_2018_paper.pdf)] [[code](https://github.com/srebuffi/residual_adapters)]

* PAD-Net: Multi-tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing (CVPR, 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_PAD-Net_Multi-Tasks_Guided_CVPR_2018_paper.pdf)]

* NestedNet: Learning Nested Sparse Structures in Deep Neural Networks (CVPR, 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kim_NestedNet_Learning_Nested_CVPR_2018_paper.pdf)]

* PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning (CVPR, 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Mallya_PackNet_Adding_Multiple_CVPR_2018_paper.pdf)] [[code](https://github.com/arunmallya/packnet)]


* **[Uncertainty]** Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR, 2018) [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)]

* Deep Asymmetric Multi-task Feature Learning (ICML, 2018) [[paper](http://proceedings.mlr.press/v80/lee18d/lee18d.pdf)]

* **[GradNorm]** GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML, 2018) [[paper](http://arxiv.org/abs/1711.02257)]

* Pseudo-task Augmentation: From Deep Multitask Learning to Intratask Sharing---and Back (ICML, 2018) [[paper](http://arxiv.org/abs/1803.04062)]

* Gradient Adversarial Training of Neural Networks (arXiv, 2018) [[paper](http://arxiv.org/abs/1806.08028)]

* Auxiliary Tasks in Multi-task Learning (arXiv, 2018) [[paper](http://arxiv.org/abs/1805.06334)]

* Routing Networks: Adaptive Selection of Non-linear Functions for Multi-Task Learning (ICLR, 2018) [[paper](http://arxiv.org/abs/1711.01239)] [[code](https://github.com/cle-ros/RoutingNetworks)

* Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering (ICLR, 2018) [[paper](http://arxiv.org/abs/1711.00108)]

### 2017

* Learning multiple visual domains with residual adapters (NeurIPS, 2017) [[paper](https://papers.nips.cc/paper/2017/file/e7b24b112a44fdd9ee93bdf998c6ca0e-Paper.pdf)] [[code](https://github.com/srebuffi/residual_adapters)]

* Learning Multiple Tasks with Multilinear Relationship Networks (NeurIPS, 2017) [[paper](https://proceedings.NeurIPS.cc/paper/2017/file/03e0704b5690a2dee1861dc3ad3316c9-Paper.pdf)] [[code](https://github.com/thuml/MTlearn)]

* Federated Multi-Task Learning (NeurIPS, 2017) [[paper](https://proceedings.NeurIPS.cc/paper/2017/file/6211080fa89981f66b1a0c9d55c61d0f-Paper.pdf)] [[code](https://github.com/gingsmith/fmtl)]

* Multi-task Self-Supervised Visual Learning (ICCV, 2017) [[paper](http://arxiv.org/abs/1708.07860)]

* Adversarial Multi-task Learning for Text Classification (ACL, 2017) [[paper](http://arxiv.org/abs/1704.05742)]

* UberNet: Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision Using Diverse Datasets and Limited Memory (CVPR, 2017) [[paper](https://arxiv.org/abs/1609.02132)]

* Fully-adaptive Feature Sharing in Multi-Task Networks with Applications in Person Attribute Classification (CVPR, 2017) [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.pdf)]

* Modular Multitask Reinforcement Learning with Policy Sketches (ICML, 2017) [[paper](http://arxiv.org/abs/1611.01796)] [[code](https://github.com/jacobandreas/psketch)]


* SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization (ICML, 2017) [[paper](http://proceedings.mlr.press/v70/kim17b.html)] [[code](https://github.com/dalgu90/splitnet-wrn)]

* One Model To Learn Them All (arXiv, 2017) [[paper](http://arxiv.org/abs/1706.05137)] [[code](https://github.com/tensorflow/tensor2tensor)]

* **[AdaLoss]** Learning Anytime Predictions in Neural Networks via Adaptive Loss Balancing (arXiv, 2017) [[paper](http://arxiv.org/abs/1708.06832)]

* Deep Multi-task Representation Learning: A Tensor Factorisation Approach (ICLR, 2017) [[paper](https://arxiv.org/abs/1605.06391)] [[code](https://github.com/wOOL/DMTRL)]

* Trace Norm Regularised Deep Multi-Task Learning (ICLR Workshop, 2017) [[paper](http://arxiv.org/abs/1606.04038)] [[code](https://github.com/wOOL/TNRDMTL)]

* When is multitask learning effective? Semantic sequence prediction under varying data conditions (EACL, 2017) [[paper](http://arxiv.org/abs/1612.02251)] [[code](https://github.com/bplank/multitasksemantics)]

* Identifying beneficial task relations for multi-task learning in deep neural networks (EACL, 2017) [[paper](http://arxiv.org/abs/1702.08303)]

* PathNet: Evolution Channels Gradient Descent in Super Neural Networks (arXiv, 2017) [[paper](http://arxiv.org/abs/1701.08734)] [[code](https://github.com/jsikyoon/pathnet)]

* Attributes for Improved Attributes: A Multi-Task Network Utilizing Implicit and Explicit Relationships for Facial Attribute Classiﬁcation (AAAI, 2017) [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14749/14282)]

### 2016 and earlier

* Learning values across many orders of magnitude (NeurIPS, 2016) [[paper](https://arxiv.org/abs/1602.07714)]

* Integrated Perception with Recurrent Multi-Task Neural Networks (NeurIPS, 2016) [[paper](https://proceedings.neurips.cc/paper/2016/file/06409663226af2f3114485aa4e0a23b4-Paper.pdf)] 

* Unifying Multi-Domain Multi-Task Learning: Tensor and Neural Network Perspectives (arXiv, 2016) [[paper](http://arxiv.org/abs/1611.09345)]

* Progressive Neural Networks (arXiv, 2016) [[paper](https://arxiv.org/abs/1606.04671)]

* Deep multi-task learning with low level tasks supervised at lower layers (ACL, 2016) [[paper](https://www.aclweb.org/anthology/P16-2038.pdf)]

* **[Cross-Stitch]** Cross-Stitch Networks for Multi-task Learning (CVPR,2016) [[paper](https://arxiv.org/abs/1604.03539)] [[code](https://github.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning)]

* Asymmetric Multi-task Learning based on Task Relatedness and Confidence (ICML, 2016) [[paper](http://proceedings.mlr.press/v48/leeb16.pdf)]

* MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving (arXiv, 2016) [[paper](http://arxiv.org/abs/1612.07695)] [[code](https://github.com/MarvinTeichmann/MultiNet)]

* A Unified Perspective on Multi-Domain and Multi-Task Learning (ICLR, 2015) [[paper](http://arxiv.org/abs/1412.7489)]

* Facial Landmark Detection by Deep Multi-task Learning (ECCV, 2014) [[paper](https://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf)] [[code](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)]

* Learning Task Grouping and Overlap in Multi-task Learning (ICML, 2012) [[paper](http://arxiv.org/abs/1206.6417)]

* Learning with Whom to Share in Multi-task Feature Learning (ICML, 2011) [[paper](http://www.cs.utexas.edu/~grauman/papers/icml2011.pdf)]

* Semi-Supervised Multi-Task Learning with Task Regularizations (ICDM, 2009) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5360282)]

* Semi-Supervised Multitask Learning (NeurIPS, 2008) [[paper](https://proceedings.neurips.cc/paper/2007/file/a34bacf839b923770b2c360eefa26748-Paper.pdf)]

* Multitask Learning (1997) [[paper](https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf)]

## [Awesome Multi-domain Multi-task Learning](https://github.com/WeiHongLee/Awesome-Multi-Domain-Multi-Task-Learning)

## Workshops

* [Workshop on Multi-Task Learning in Computer Vision (DeepMTL) at ICCV 2021](https://sites.google.com/view/deepmtlworkshop/home)

* [Adaptive and Multitask Learning: Algorithms & Systems Workshop (AMTL) at ICML 2019](https://www.amtl-workshop.org)

* [Workshop on Multi-Task and Lifelong Reinforcement Learning at ICML 2015](https://sites.google.com/view/mtlrl)

* [Transfer and Multi-Task Learning: Trends and New Perspectives at NeurIPS 2015](https://nips.cc/Conferences/2015/Schedule?showEvent=4939)

* [Second Workshop on Transfer and Multi-task Learning at NeurIPS 2014](https://sites.google.com/site/multitaskwsnips2014/)

* [New Directions in Transfer and Multi-Task: Learning Across Domains and Tasks Workshop at NeurIPS 2013](https://sites.google.com/site/learningacross/home)

## Online Courses

* [CS 330: Deep Multi-Task and Meta Learning](https://cs330.stanford.edu)

## Related awesome list

* https://github.com/SimonVandenhende/Awesome-Multi-Task-Learning

* https://github.com/Manchery/awesome-multi-task-learning



