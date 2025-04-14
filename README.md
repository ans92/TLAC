# TLAC: Two-stage LMM Augmented CLIP for Zero-Shot Classification [CVPRW 2025]

**[TLAC: Two-stage LMM Augmented CLIP for Zero-Shot Classification](https://arxiv.org/pdf/2503.12206)**

[Ans Munir](https://scholar.google.com/citations?user=AdQOl2IAAAAJ&hl=en), [Faisal Z. Qureshi](https://vclab.ca/), [Muhammad Haris Khan](https://m-haris-khan.com/), [Mohsen Ali](https://mohsenali.github.io/)

_______

## Highlights

> <div align="justify"> <strong>Abstract:</strong> Contrastive Language-Image Pretraining (CLIP) has shown impressive zero-shot performance on image classification. However, state-of-the-art methods often rely on fine-tuning techniques like prompt learning and adapter-based tuning to optimize CLIP's performance. The necessity for fine-tuning significantly limits CLIP's adaptability to novel datasets and domains. This requirement mandates substantial time and computational resources for each new dataset. To overcome this limitation, we introduce simple yet effective training-free approaches, Single-stage LMM Augmented CLIP (SLAC) and Two-stage LMM Augmented CLIP (TLAC), that leverages powerful Large Multimodal Models (LMMs), such as Gemini, for image classification.  The proposed methods leverages the capabilities of pre-trained LMMs, allowing for seamless adaptation to diverse datasets and domains without the need for additional training. Our approaches involve prompting the LMM to identify objects within an image. Subsequently, the CLIP text encoder determines the image class by identifying the dataset class with the highest semantic similarity to the LLM predicted object. Our models achieved superior accuracy on 9 of 11 base-to-novel datasets, including ImageNet, SUN397, and Caltech101, while maintaining a strictly training-free paradigm.  Our TLAC model achieved an overall accuracy of 83.44%, surpassing the previous state-of-the-art few-shot methods by a margin of 6.75%. Compared to other training-free approaches, our TLAC method achieved 83.6% average accuracy across 13 datasets, a 9.7% improvement over the previous methods.
</div>

## Main Contributions

* We propose a simple yet effective training-free approach to leverage Large Multimodal Models (LMMs) for image classification task.
* Specifically, this work introduces two models, Singlestage LMM Augmented CLIP (SLAC) and Two-stage LMM Augmented CLIP (TLAC), which combine the strengths of LMM and VLM to leverage their combined capabilities.
* Our experiments demonstrate that our approach achieves superior accuracy on a majority of evaluated datasets, including the large-scale ImageNet, all while remaining entirely training-free and requiring no training samples.

## Results

### Training-Free methods

Table shows the Average Accuracy of Training-Free methods over 13 datasets. Higher results are better and shown in bold.

| Name  | Average Accuracy |
| ------------- | ------------- |
| CLIP-S   | 69.9  |
| CLIP-DS  | 71.6  |
| CuPL  | 75.2  |
| D-CLIP    | 71.7  |
| Waffle  | 71.7  |
| MPVR (Mix)  | 72.9  |
| MPVR (GPT)  | 73.9  |
| Ours (SLAC)  | 79.4  |
| **Ours (TLAC)**  |  **83.6** |

### Base-to-Novel Generalization methods

Table shows the Average Novel Accuracy of few-shot methods over 11 datasets. Higher results are better and shown in bold.

| Name  | Average Novel Accuracy |
| ------------- | ------------- |
| CLIP  | 74.22  |
| CoOp  | 63.22  |
| Co-CoOp   | 71.69  |
| ProDA   | 72.30  |
| KgCoOp  |  73.60 |
| MaPLe  | 75.14  |
| LASP  |  74.90 |
| RPO  | 75.00  |
| MMA  |  76.80 |
| Ours (SLAC)  | 78.69  |
| **Ours (TLAC)**  |  **83.44** |

## Citation

If you find this work helpful for your research, please consider citing:

```
@article{munir2025tlac,
  title={TLAC: Two-stage LMM Augmented CLIP for Zero-Shot Classification},
  author={Munir, Ans and Qureshi, Faisal Z and Khan, Muhammad Haris and Ali, Mohsen},
  journal={arXiv preprint arXiv:2503.12206},
  year={2025}
}
```
