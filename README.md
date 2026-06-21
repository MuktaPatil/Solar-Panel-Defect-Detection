# Cross-Modal Transfer Learning for Solar Panel Defect Detection

Solar panels degrade silently - microcracks, hotspots, and electrical failures reduce output long before they're visible to the naked eye. This project builds an end-to-end AI system for **holistic solar panel health monitoring** across both stages of a panel's life: factory inspection and field deployment.

We trained high-accuracy defect classifiers on two complementary imaging modalities : **Electroluminescence (EL)** imaging for microscopic structural defects during manufacturing, and **Thermal Infrared** imaging for operational heat anomalies in the field - then scientifically investigated whether a unified cross-modal model could serve both, and *why* each modality captures fundamentally different information about panel health.

*Skills: Transfer Learning · Computer Vision · EfficientNet · PyTorch · Data Augmentation · Imbalanced Classification · Grad-CAM · Statistical Validation*

---

## What We Built

Two high-performance defect detection classifiers - one for thermal imaging, one for EL imaging - then systematically tested cross-modal transfer in both directions:

- **Thermal → EL transfer**: Did heat-pattern features help detect structural cracks?
- **EL → Thermal transfer**: Did structural features improve heat anomaly detection?

---

## Results

| Task | Model | Test F1-Score |
|---|---|---|
| Thermal Defect Classification | EfficientNet-B3 | **95.98%** |
| EL Defect Classification | EfficientNet-B3 (matched aug) | **90.79%** |
| Thermal → EL Transfer | EfficientNet-B3 | -3.44% vs baseline |
| EL → Thermal Transfer | EfficientNet-B3 | -0.38% vs baseline |

Both classifiers exceeded their performance targets. Cross-modal transfer, however, showed **limited and asymmetric benefit** — transferring thermal features to EL imaging actively hurt performance, while the reverse direction nearly matched baseline with a large enough model.

---

## Why the Asymmetry?

EL images encode **universal structural information** — cell boundaries, grain patterns, microscopic cracks — that partially generalizes to thermal images.

Thermal images encode **heat distribution patterns** that are consequences of defects, not the defects themselves. These patterns are modality-specific and do not generalize to structural defect detection.

This difference in information abstraction level drives the 3.06 percentage point asymmetry between transfer directions.

---

## Datasets

| Dataset | Modality | Images | Classes |
|---|---|---|---|
| PV-Multi-Defect | Thermal Infrared | 1,106 | 6 (hotspot, scratch, black border, no electricity, broken, normal) |
| ELPV | Electroluminescence | 2,624 | 2 (functional, defective) |

Both datasets exhibited class imbalance, handled via weighted random sampling and class-weighted loss functions.

---

## Tech Stack

- **Model:** EfficientNet-B0, B1, B3 (PyTorch)
- **Augmentation:** Albumentations + torchvision transforms
- **Explainability:** Grad-CAM visualizations
- **Training:** Adam optimizer, ReduceLROnPlateau scheduler, early stopping
- **Compute:** ~80 hours on Tesla T4 GPU

---

## Key Takeaways

- Modern CNNs can detect solar panel defects with high accuracy even on small datasets
- Cross-modal transfer learning is **not a silver bullet** — domain gap must be measured, not assumed
- Augmentation strategy alone improved ELPV baseline by **+5.04% F1** — often underestimated
- The industry practice of maintaining separate thermal and EL inspection pipelines is statistically validated

---

## Future Directions

- Multimodal fusion architectures (early/late fusion of both imaging streams)
- Adversarial domain adaptation (DANN) to close the thermal-EL domain gap
- Vision Transformers (ViT) as an alternative backbone for cross-modal tasks
- Self-supervised pre-training on unlabeled solar panel images

  ## Files

Data Link: https://drive.google.com/drive/folders/1aBhbZr8Sj3FqnuqYaJXMULHzFKrcWlJY?usp=sharing
Report: Solar Panel Report.docx
results stored in the folder: Images, baseline jsons, confusion matrix, training and testing accuracy charts and curves.
