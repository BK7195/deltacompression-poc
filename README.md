# Delta Compression (Proof of Concept)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BK7195/delta-compression-poc/blob/main/run_inference.ipynb)

This is the official repository for the Proof of Concept (PoC) of the paper "Delta Compression: Towards Efficient Semantic Compression via Hypernetwork-Generated Parameter Deltas".

[https://zenodo.org/records/15876304]

## Overview

[cite_start]This repository contains the LoRA adapter, the dataset, and the inference script to reproduce the results presented in the paper.  [cite_start]The experiment demonstrates that a minimal LoRA adapter (approx. 390KB) can store and perfectly reconstruct 70,000 tokens of diverse Japanese texts. 

## How to Run (on Google Colab)

This project is designed and verified to run on Google Colab, ensuring easy and accessible reproducibility.

1.  **Click the "Open in Colab" badge** at the top of this page.
2.  A new notebook will open in Google Colab.
3.  Run the cells sequentially from top to bottom to see the results. The notebook will automatically clone this repository, install dependencies, and run the inference to generate the 70 texts.

## File Structure

├── adapter/          # LoRA adapter files 
├── data/             # Dataset files (70 texts) 
├── inference.py      # The original script to run inference 
├── run_inference.ipynb # Notebook for easy execution on Colab
└── requirements.txt  # Required Python libraries

## License
This project is licensed under the Apache License 2.0. 