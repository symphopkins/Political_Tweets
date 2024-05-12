# Political Tweets: Twitter Data Analysis

## Overview
This repository contains code for performing sentiment analysis on Twitter data related to political candidates. The dataset includes tweets made by people, mainly focused on tweets made by people on Modi (2019 Indian Prime Minister candidate) and other prime ministerial candidates.

## Files Included
- `Political_Tweets.py`: Python script containing the code.
- `Political_Tweets.ipynb`: Google Colab Notebook containing the detailed code implementation and explanations.
- `requirements.txt`: Text file listing the required Python packages and their versions.
- `LICENSE.txt`: Text file containing the license information for the project.

## Installation
To run this project, ensure you have Python installed on your system. You can install the required dependencies using the `requirements.txt` file.

## Usage
1. Load the dataset into memory using the pandas library.
2. Convert the column `clean_text` to a matrix of token counts using CountVectorizer and unigrams and bigrams.
3. Perform the tf-idf analysis on the column `clean_text` using CountVectorizer and TfidfTransformer.
4. Perform the tf-idf analysis on the column `clean_text` using TfidfVectorizer.
5. Perform the tf-idf analysis on the column `clean_text` using HashingVectorizer and TfidfTransformer.

## Data Source
The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset/data).

## License
 Creative Commons Attribution 4.0 International License
