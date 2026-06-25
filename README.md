# Disaster Tweets Project

Binary text classification project for the Kaggle competition `nlp-getting-started`.

## Goal

Predict whether a tweet is about a real disaster.

- `1` = disaster tweet
- `0` = non-disaster tweet

The project is designed as a clear, presentation-friendly NLP workflow with explicit preprocessing, classical baselines, and transformer experiments.

The repository also contains two API-based LLM experiments:

- a Groq-based prompt classification notebook
- a Gemini-based prompt classification notebook

Those experiments were useful for comparison, but free-tier token and request limits made them unstable for a full run.

## Data

The repository contains:

- `data/train.csv` for training and validation
- `data/test.csv` for final prediction generation

Important columns:

- `text`
- `target`

## Main dataset issue

The training set contains duplicate tweets with conflicting labels.

This project explicitly:

- detects conflicting duplicate tweet texts
- removes all rows belonging to those conflicting texts for the main modeling run
- reports the number of conflicts in the notebook and project context

This matters because noisy labels can distort both model training and evaluation.

## Preprocessing

The notebook uses a transparent preprocessing pipeline written as separate steps:

1. convert text to lowercase
2. remove URLs
3. remove punctuation, digits, and non-alphabetic characters
4. tokenize the text
5. remove stop words
6. apply a lightweight lemmatization step
7. join cleaned tokens back into a single string

This keeps the classical baseline easy to explain in a presentation.

## Models

The project compares:

- Logistic Regression with TF-IDF
- Linear SVM with TF-IDF
- Random Forest with TF-IDF
- a fast `roberta-base` experiment on a single train/validation split
- a stronger RoBERTa v2 / v3 tuning path with softer text cleaning, longer context, and weighted loss
- an active `microsoft/deberta-v3-base` sanity run with soft cleaning and keyword context
- Groq and Gemini prompt-based classification experiments

## Evaluation

The main metric is `F1`.

The notebook also reports:

- classification report
- confusion matrix
- best validation threshold for the transformer run

## Current results

Validation results obtained so far:

- Logistic Regression baseline: `F1 = 0.7873`
- Linear SVM baseline: `F1 = 0.7695`
- Random Forest baseline: `F1 = 0.7433`
- Fast RoBERTa validation run: `F1 = 0.7932`
- RoBERTa v2 validation run: `F1 = 0.8116`
- RoBERTa v3 validation run: `F1 = 0.8049`
- DeBERTa v3 base validation run: `F1 = 0.8245`
- Best RoBERTa threshold: `0.35`
- Best DeBERTa threshold: `0.300`

DeBERTa v3 base is currently the strongest completed model in this project run.

## LLM experiments

Groq notebook findings:

- prompt-based classification worked
- validation F1 on a small split was much lower than RoBERTa
- the free tier exposed strict TPM and request rate limits
- batched prompts often had to be split automatically to avoid quota errors
- the final result was unstable and not competitive with RoBERTa

Gemini notebook findings:

- model access was available
- the notebook repeatedly hit free-tier request quotas and output/response issues
- the free-tier limits made the experiment impractical for a full validation run
- the final result was not useful as the main model for the project

Presentation takeaway:

- API-based LLMs are a valid experiment for comparison
- for this dataset, local fine-tuned RoBERTa was more reliable, reproducible, and practical
- DeBERTa-v3-base is the strongest completed transformer result and the best model in the project so far

## Project structure

- `disaster_tweets_project.ipynb` - main notebook with data inspection, preprocessing, baselines, and transformer experiment
- `project_context.md` - project brief and implementation notes
- `data/train.csv` - labeled training data
- `data/test.csv` - unlabeled Kaggle test data

## How to run

1. Open `disaster_tweets_project.ipynb`
2. Run the data loading and conflict detection cells
3. Run the preprocessing and baseline model cells
4. Install transformer dependencies if needed
5. Run the fast RoBERTa experiment cell

If you want to generate a Kaggle submission, use the transformer output or the chosen model's predictions on `test.csv` and save them in the Kaggle submission format:

- `id`
- `target`

## Notes

- The notebook is intentionally structured for a short class presentation.
- The project favors explainable preprocessing and readable baselines before trying stronger transformer models.
- The current RoBERTa score is a strong validation result, especially given noisy duplicate labels in the dataset.
- LLM API experiments are documented for completeness, but they were constrained by free-tier limits and are not the main result.
