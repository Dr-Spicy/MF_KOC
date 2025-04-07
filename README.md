# MF_KOC
KOC based cyber campaign on Xiaohongshu(Rednote) for MeetFresh DFW 

## 1. How to collab on this repo (branch, commit, migrate etc.)
Please read this [document](./Docs/Repo_Collab_Guide.md)!!!!

## 2. Environment set up
Please refer to this [yml file](./Docs/mf_koc(py39).yaml) to set up a new AutoGluon and PyTorch compatible virtual environment for this project. It is mainly based on:

    - python=3.9.21
    - pandas=2.2.2
    - numpy=1.24.4
    - torch=2.3.1+cu118
    - transformers=4.39.3

## 3. File structure (update if you add items)

    /Data                   - All data
     ├── raw                - Raw creators and contents JSON files from MediaCrawler
     └── processed          - Combined and preprocessed based on the EDA description
        ├── contents_cooked.json   - Processed contents data
        ├── contents_raw.json      - Raw contents data
        ├── creator_cooked.json    - Processed creators data
        └── creator_raw.json        - Raw creators data
    /DataPreprocessing
     ├── preprocessing.ipynb    - Code for creators and contents data preprocessing & feature engineering
    /Docs                   - All documents, including Google Drive files and other local ones
     ├── GoogleDriveFiles   - Links to the Google Drive Folder
     ├── Repo_Collab_Guide  - How to on branches
     ├── ListOfCompetitors  - Yelp comparison
     ├── 5打分维度           - 打分依据
     ├── 
     └── mf_koc(py39)       - Virtual Environment config

    /EDA
     ├── RadarChart         - Function to plot radar chart
     ├── creator_and_content_tables         - Creator and Content Tables
     ├── day_count & week_count          - Function to plot interaction bar plot by time

    /Figs                   - All Figures

    /Results                - All reports and results

    /src                    - Migrated repos from outside
