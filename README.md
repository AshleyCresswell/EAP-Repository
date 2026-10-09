# Measuring Expressive Affective Polarisation in Online Political Communication

This repository contains the computational pipeline for a study of **Expressive Affective Polarisation (EAP)** in online political communication during the 2024 U.S. presidential election campaign.

Using YouTube comments from the official campaign channels of Donald Trump, Joe Biden, and Kamala Harris, the study develops and applies computational measures of affective polarisation and political incivility to examine how hostile political expression varies across an election campaign and around major political events.

## Research Overview

Affective polarisation is traditionally measured using survey-based evaluations of political ingroups and outgroups. Measuring it through online political communication presents a different challenge: observable negativity or hostility does not necessarily constitute evidence of affective polarisation.

This project therefore develops **Expressive Affective Polarisation (EAP)** as a multidimensional approach to identifying expressions of affective polarisation in online political comments. EAP is operationalised through three dimensions: **Aversion, Othering, and Social Distance**.

These measures are benchmarked against a broader measure of **political incivility**, allowing the analysis to distinguish expressions of affective polarisation from more general forms of hostile or uncivil political communication.

## Data

The analysis uses English-language YouTube comments collected from the official campaign channels of Donald Trump, Joe Biden, and Kamala Harris between **8 March and 5 November 2024**.

Following eligibility filtering and duplicate removal, the final corpus contains **576,977 comments across 742 campaign videos**.

Raw YouTube comments are not redistributed through this repository.

## Methodology

The project combines manual annotation, supervised natural language processing, and temporal analysis.

### 1. Operationalisation

Separate operationalisation frameworks were developed for **Expressive Affective Polarisation (EAP)** and **political incivility**. The EAP framework identifies three theoretically distinct forms of expression:

- **Aversion** — expressions of dislike or negative affect towards political outgroups.
- **Othering** — expressions that differentiate or delegitimise political outgroups.
- **Social Distance** — expressions indicating unwillingness to associate with political outgroups.

The incivility framework provides a broader benchmark for identifying hostile and uncivil political expression.

### 2. Annotation and Classification

Comments were manually annotated using these frameworks. Separate BERT classifiers were trained to identify the three EAP dimensions, evaluated on held-out data, and then applied to the full comment corpus.

### 3. Analysis

Comment-level predictions were aggregated across the campaign to examine the prevalence of EAP and incivility, temporal trends, and changes surrounding major campaign events.

## Repository Structure

```text
src/
├── 01_collect_data.py
├── 02_prepare_corpus.py
├── 03_prepare_training_data.py
├── 04_train_models.py
├── 05_evaluate_models.py
├── 06_predict_corpus.py
└── 07_analysis.py
```

The pipeline covers:

**Data collection → Corpus construction → Annotation preparation → Model training → Model evaluation → Full-corpus prediction → Statistical analysis**

## Key Findings

EAP constitutes a substantially narrower form of political expression than general incivility. Approximately **8.1% of comments contained EAP**, compared with approximately **49.9% classified as politically uncivil**.

The analysis also finds that hostile political expression does not simply increase as electoral competition intensifies. Instead, EAP and incivility respond differently to major campaign events, suggesting that changes in online political hostility are contingent on the political context.

## Reproduction

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Scripts are designed to be run sequentially from `01_collect_data.py` through `07_analysis.py`.

YouTube collection requires a YouTube Data API key supplied through an environment variable. Large raw datasets and trained model files are not included in the repository.

## Citation

This repository accompanies ongoing doctoral research on affective polarisation and online political communication. A full citation will be added following publication.
