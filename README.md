# Radiology-RAG
## LMA Major Project - Radiology Imaging Techniques
## Team OppenAImer - Naman Mishra and Vempaty Saket

---

- Report is `Report.pdf` and is also present [here](https://docs.google.com/document/d/1DpEWQFicDV2Sv40Yv-sWva3gN2DVTwh7Mo1VbSQBqEM/edit?usp=sharing)

- Demo videos are present [here](https://drive.google.com/drive/folders/1DqVLOEQUnSfOC0RP3cE7qA475_RhymhN?usp=sharing)

---

## Repository Contents

- `assets` : Contains example images, crafted prompts, sample responses, etc.
- `configs` : .yaml files storing pipeline hyperparameters.
- `radrag` : Contains the main source code for RAG-enhanced radiology applications.
- `utils` : Constains scripts used for generating vector databases for both images and text.
- `environment.yml` : Conda configuration for installation.
- `report_generation.py` : Report-generation application.

---

## Installation

```git clone https://github.com/ltrc/lma-major-project-oppenaimer.git```

```cd lma-major-project-oppenaimer```

```conda env create -f environment.yml```

---

## Running Applications

### Report Generation

```streamlit run report_generation.py --server.runOnSave true```

### Quiz Generation/ Textbook/Web Search 

```streamlit run router_v9.py --server.runOnSave true```
