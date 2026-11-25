# AdikeRakshak AI - Offline ARM Pipeline

**Arecanut Leaf and Lesion Isolation, On-Device Disease Classification, and Guarded Farmer Advisory**

- **Authors / Team:** Beauty Sahani, Daneshwari Hiremath, Aman Singh (ECE Department, Atria Institute of Technology)
- **Deployment Target:** Raspberry Pi 5 / Linux-based ARM64 Device (Offline-First Mode)
- **Date & Status:** November 25, 2026

## Overview

AdikeRakshak AI is an offline-first smart agriculture decision-support system for arecanut plantations where internet connectivity is weak or unavailable. Farmers can upload or capture plant images via a Streamlit interface to receive on-device disease classification, visual symptom estimates, and guarded agronomic guidance.

## Dataset & Model Summary

- **Dataset:** 8,847 images across 9 classes (7,078 train / 1,769 validation split + 2,216 test folder images):
  - Mahali / Koleroga: 2,563 (28.97%)
  - Healthy Nut: 1,886 (21.32%)
  - Yellow Leaf Disease: 1,477 (16.70%)
  - Healthy Trunk: 1,432 (16.19%)
  - Healthy Leaf: 604 (6.83%)
  - Stem Cracking: 540 (6.10%)
  - Stem Bleeding: 151 (1.71%)
  - Bud Borer: 144 (1.63%)
  - Healthy Foot: 50 (0.57%)
- **Vision Model:** MobileNetV2 (224x224 RGB, input_layer_1 -> dense), Keras model ~9.77 MB converted to ONNX ~8.54 MB (CPUExecutionProvider), dashboard-reported overall accuracy **99.06%** (precision **99.04%**, recall **99.04%**, F1-score **99.04%**).
- **Advisory Layer:** Combines verified local agronomy reference data with Llama 3.2 1B (llama3.2:1b via Ollama / ExecuTorch) and deterministic guardrails.
- **Deployment Status:** Keras -> ONNX structural check and single-image parity verified; U2-NetP segmentation, INT8/XNNPACK quantization, ARM latency benchmarks, and ExecuTorch .pte validation are planned/pending.

## Project Report

See [AdikeRakshak_Offline_ARM_Pipeline_Project_Report.pdf](./AdikeRakshak_Offline_ARM_Pipeline_Project_Report.pdf) for full methodology, limitations, and future work.
