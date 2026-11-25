# AdikeRakshak AI - Offline ARM Pipeline

**Arecanut Leaf and Lesion Isolation, On-Device Disease Classification, and Guarded Farmer Advisory**

- **Team:** Beauty Sahani, Daneshwari Hiremath, Aman Singh
- **Target Hardware:** Raspberry Pi 5 / Linux-based ARM device (Offline-First Goal)
- **Date & Status:** 25/11/2026

## Overview & Methodology

Offline-first smart agriculture system for arecanut crop disease detection (MobileNetV2 transfer learning, ONNX export, Streamlit UI, and guarded farmer advisory) with planned leaf/lesion isolation, ARM quantization/benchmarking, and offline ExecuTorch runtime validation.

## Dataset Summary (8,847 Total Images)

- **Split:** 7,078 training, 1,769 validation, plus a separate test folder with 2,216 images.
- **Classes:** Healthy Leaf (604), Healthy Nut (1,886), Healthy Trunk (1,432), Mahali / Koleroga (2,563), Stem Bleeding (151), Bud Borer (144), Healthy Foot (50), Stem Cracking (540), Yellow Leaf Disease (1,477).

## Model Conversion & Reported Metrics

- **Vision Backbone:** MobileNetV2 (224x224 input). Keras (~9.77 MB) converted to ONNX (~8.54 MB) with matching single-image prediction.
- **Reported Metrics:** Overall accuracy 99.06%, precision 99.04%, sensitivity/recall 99.04%, F1-score 99.04% (~99.01% test accuracy noted in separate training record).

## Final Report

See AdikeRakshak_Offline_ARM_Pipeline_Project_Report.pdf in this repository.
