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

<table>
  <tr>
    <td valign="top" width="50%">
      <h3>Training-Free methods</h3>
      <p>Table shows the Average Accuracy of Training-Free methods over 13 datasets. Higher results are better and shown in bold.</p>

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Average Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CLIP-S</td>
      <td>69.9</td>
    </tr>
    <tr>
      <td>CLIP-DS</td>
      <td>71.6</td>
    </tr>
    <tr>
      <td>CuPL</td>
      <td>75.2</td>
    </tr>
    <tr>
      <td>D-CLIP</td>
      <td>71.7</td>
    </tr>
    <tr>
      <td>Waffle</td>
      <td>71.7</td>
    </tr>
    <tr>
      <td>MPVR (Mix)</td>
      <td>72.9</td>
    </tr>
    <tr>
      <td>MPVR (GPT)</td>
      <td>73.9</td>
    </tr>
    <tr>
      <td>Ours (SLAC)</td>
      <td>79.4</td>
    </tr>
    <tr>
      <td><strong>Ours (TLAC)</strong></td>
      <td><strong>83.6</strong></td>
    </tr>
  </tbody>
</table>
    </td>
    <td valign="top" width="50%">
      <h3>Base-to-Novel Generalization methods</h3>
      <p>Table shows the Average Novel Accuracy of few-shot methods over 11 datasets. Higher results are better and shown in bold.</p>
      <table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Average Novel Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CLIP</td>
      <td>74.22</td>
    </tr>
    <tr>
      <td>CoOp</td>
      <td>63.22</td>
    </tr>
    <tr>
      <td>Co-CoOp</td>
      <td>71.69</td>
    </tr>
    <tr>
      <td>ProDA</td>
      <td>72.30</td>
    </tr>
    <tr>
      <td>KgCoOp</td>
      <td>73.60</td>
    </tr>
    <tr>
      <td>MaPLe</td>
      <td>75.14</td>
    </tr>
    <tr>
      <td>LASP</td>
      <td>74.90</td>
    </tr>
    <tr>
      <td>RPO</td>
      <td>75.00</td>
    </tr>
    <tr>
      <td>MMA</td>
      <td>76.80</td>
    </tr>
    <tr>
      <td>Ours (SLAC)</td>
      <td>78.69</td>
    </tr>
    <tr>
      <td><strong>Ours (TLAC)</strong></td>
      <td><strong>83.44</strong></td>
    </tr>
  </tbody>
</table>
    </td>
  </tr>
</table>

## Environment setup and code installation

For setting up the environment and code, please follow the instructions in [INSTALL.md](https://github.com/ans92/TLAC/blob/main/docs/INSTALL.md).

## Dataset preparation

To prepare the datasets, please follow the instruction in [DATASETS.md](https://github.com/ans92/TLAC/blob/main/docs/DATASETS.md)

## Model Running

To run the model, please follow the instructions in [RUN.md](https://github.com/ans92/TLAC/blob/main/docs/RUN.md)

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

## Contact

If you have any questions, please create an issue on this repository or contact at [msds20033@itu.edu.pk](mailto:msds20033@itu.edu.pk)

## Acknowledgements

This code is based on [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning/tree/main). We thank the authors for their work. If you use our code, please also consider citing their work as well.
