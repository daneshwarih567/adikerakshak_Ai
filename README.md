# AdikeRakshak AI - Offline ARM Pipeline

**Arecanut Leaf and Lesion Isolation, On-Device Disease Classification, and Guarded Farmer Advisory**

- **Team:** Beauty Sahani, Daneshwari Hiremath, Aman Singh
- **Target Hardware:** Raspberry Pi 5 / Linux-based ARM device (Offline-First Goal)
- **Date & Status:** 25/11/2026

## Overview & Methodology

AdikeRakshak AI is designed with an offline-first deployment goal for farms with weak or unavailable internet connectivity. Through a Streamlit interface, a farmer can upload or capture an arecanut image to obtain a predicted class and confidence score, along with planned extensions for leaf and lesion isolation, on-device inference, post-processing, locally retrieved guarded farmer guidance, and stage-wise latency profiling.

## Dataset Summary (8,847 Total Images)

Overall recorded split: **7,078 training** images and **1,769 validation** images, plus a separate test folder containing **2,216** images (per-class train/validation counts are not available in the supplied records).

| Class Name | Total Samples | Class Share |
| :--- | :---: | :---: |
| Healthy Leaf | 604 | 6.83% |
| Healthy Nut | 1,886 | 21.32% |
| Healthy Trunk | 1,432 | 16.19% |
| Mahali / Koleroga | 2,563 | 28.97% |
| Stem Bleeding | 151 | 1.71% |
| Bud Borer | 144 | 1.63% |
| Healthy Foot | 50 | 0.57% |
| Stem Cracking | 540 | 6.10% |
| Yellow Leaf Disease | 1,477 | 16.70% |
| **TOTAL** | **8,847** | **100.00%** |

## Model Conversion, Optimization & Reported Metrics

- **Vision Backbone:** MobileNetV2 transfer learning with `224x224` input size (`input_layer_1` -> `dense`).
- **Artifacts & Conversion:** Original TensorFlow/Keras model (~**9.77 MB**) converted to ONNX (~**8.54 MB**, ~12.6% smaller) using `CPUExecutionProvider`. ONNX structural check passed and a single-image test produced matching prediction index 5 (~`99.79845%` confidence) across Keras and ONNX.
- **Dashboard-Reported Metrics:** Overall accuracy **99.06%**, precision **99.04%**, sensitivity/recall **99.04%**, F1-score **99.04%** (separate training record notes ~99.01% test accuracy for the original MobileNetV2 model).

## Implementation & Validation Status

- **Implemented / Basic Check Done:** Streamlit image upload & camera-capture interface, MobileNetV2 classifier, and 8.54 MB ONNX model (structural check + one-image parity verified).
- **Pending / Planned Validation:** Per-class split counts, full held-out test set evaluation, U2-NetP leaf/lesion isolation integration, INT8 + XNNPACK quantization & ARM latency/p95 benchmarks, and complete offline ExecuTorch Llama (.pte) runtime validation.

## Project Documentation & Final Report

- **Final Report:** [`AdikeRakshak_Offline_ARM_Pipeline_Project_Report.pdf`](./AdikeRakshak_Offline_ARM_Pipeline_Project_Report.pdf)
- **Repository Link:** `https://github.com/daneshwarih567/adikerakshak_Ai`
