# Skin Concern Advisor

ISOM5240 Group Project — Sephora Skin Concern Advisor

## Overview

A two-pipeline deep learning app:

1. **ViT** (fine-tuned on ACNE04-v2) — classifies selfie acne severity
2. **DistilBERT** (pre-trained SST-2) — classifies buyer reviews into PRO / CON

## Model

- Vision: [jiefangziyou/vit-acne-severity](https://huggingface.co/jiefangziyou/vit-acne-severity)
- Review: distilbert-base-uncased-finetuned-sst-2-english

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
