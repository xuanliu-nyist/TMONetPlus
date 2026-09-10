# TMO-Net+: An Enhanced Tumor Multi-Omics Pre-Trained Network for Multi-Task Learning in Oncology
Liu, W.; Liu, X.; Zhou, S.; Li, K.; Wang, X.; Chen, K.; Guo, L.; Zhang, R.; Su, Q. TMO-Net+: An Enhanced Tumor Multi-Omics Pre-Trained Network for Multi-Task Learning in Oncology. Genes 2026, 17, 1085. https://doi.org/10.3390/genes17091085
# Introduction
TMO‑Net+ is an enhanced pre‑trained deep learning model for tumor multi‑omics data, designed to learn more robust and cross‑cancer transferable multi‑omics representations. It aims to improve performance on downstream oncology tasks such as pan‑cancer classification, primary/metastatic site prediction, and prognostic modeling.

TMO‑Net+ is an improved and extended version built upon TMO‑Net (Wang et al., Genome Biology, 2024). By introducing a feature attention encoder, a gated Mixture‑of‑Experts (MoE) module, and a supervised classification loss, it effectively alleviates practical challenges including missing modalities, incomplete within‑omics data, and high‑dimensional noise.

# Method
TMO-Net+ introduces three key improvements over the original TMO-Net (Wang et al., 2024):

1、Feature Attention Encoder

For each omics modality, a Multilayer Perceptron (MLP) is used to extract latent features, and a gated attention mechanism is introduced at the end of the encoder. This mechanism projects the original input through a learnable sigmoid gate and performs element-wise multiplication with the deep features, thereby recalibrating the feature representations and effectively suppressing modality-dependent input variation and redundant information.

2、Product-of-Experts (PoE) + Gated Mixture-of-Experts (MoE) Fusion Module

PoE multiplicatively fuses the posterior distributions (mean and variance) output by each modality encoder to obtain a cross-modal consensus representation. For missing modalities, the PoE fusion result is directly used as their latent representation.

MoE consists of multiple expert networks and a centralized gating network. Each expert applies a nonlinear transformation to the single-modality latent representation, while the gating network generates normalized modality contribution weights for each sample based on the global context, enabling adaptive weighted fusion that is finally concatenated into a joint embedding.

3、Supervised Classification Head and Tailored Loss Function

During the pre-training stage, a deep classification head is added to the model, and both a contrastive loss (pulling same-class samples together and pushing different-class samples apart, with Euclidean or cosine distance as the distance metric) and a cross-entropy classification loss are optimized simultaneously. These two losses jointly guide the latent space to form a more discriminative inter-class separation structure.
<img width="1490" height="678" alt="network" src="https://github.com/user-attachments/assets/db70ba97-5df5-41eb-b1de-4e2a6afc0ecb" />

# Dataset and Data pre-processing

The multi-omics data used for pre-training and downstream tasks were all downloaded from public sources. The processed data are available at 
(https://www.kaggle.com/datasets/xiaozhuang1/tmonetplus)


# Acknowledgements
We sincerely thank the authors of TMO-Net (Wang et al., Genome Biology, 2024) for openly releasing their pre-trained model, code, and multi-omics data processing pipeline. TMO-Net provided an important foundation and architectural reference for this work, enabling the development and downstream evaluation of TMO-Net+. The open-source code is available at https://github.com/FengAoWang/TMO-Net. We also acknowledge TCGA and other public data resources, as well as the open-source tools used in this study.
