# Disaster Tweets Project Context

## Goal
Build a binary text classifier for the Kaggle competition `nlp-getting-started` that predicts whether a tweet is about a real disaster:

- `1` = disaster tweet
- `0` = non-disaster tweet

The result will be presented in a class presentation, so the notebook should be clear, structured, and easy to explain.

## Data

- Main training file: `data/train.csv`
- Use only `train.csv` for modeling because `test.csv` has no labels.
- Important columns:
  - `text`
  - `target`

## Important dataset issue

There may be duplicate tweets with different labels.
This must be checked explicitly in the notebook, because the training set can be dirty and this can affect model quality and evaluation.

In practice, the safest default for the project is:

- detect those conflicting duplicate texts
- remove them from the training set for the main modeling run
- mention the number of conflicts in the notebook and presentation

## Project expectations

The notebook should include:

1. Data loading and inspection
2. Explicit duplicate/conflict check
3. Text preprocessing
4. Feature extraction
5. Model training
6. Evaluation with `F1`
7. Comparison of several models
8. Final interpretation and limitations

Also include a lightweight comparison to prompt-based LLM API experiments, but keep them separate from the main modeling path.

## Preprocessing guidance

- lowercase text
- remove punctuation and non-alphabetic tokens
- tokenize
- remove stop words
- lemmatize if possible

## Modeling guidance

Start with simple and explainable baselines, then move to the strongest transformer model for the final run:

- Logistic Regression
- SVM
- Random Forest
- XGBoost if available
- pretrained transformer fine-tuning with Hugging Face

Recommended final model family:

- `microsoft/deberta-v3-base` if the environment supports it
- fallback to `roberta-base` if memory is tighter

For this project, RoBERTa was more practical than API-based LLMs because the free-tier limits on Groq and Gemini made full validation runs unstable.

The `microsoft/deberta-v3-base` sanity run is now the strongest completed transformer result, and it used soft preprocessing and keyword context as a single-split check before any 5-fold CV.

Recommended project strategy:

- use `clean_train_df` after removing conflicting duplicate texts
- run 5-fold Stratified K-Fold cross-validation
- save the best fold checkpoint by F1
- average test probabilities across folds
- tune the final threshold on OOF probabilities

## Evaluation

Focus on:

- `F1` score
- classification report
- confusion matrix

For the final report, include a summary table of all tried models and their parameters, plus the best score obtained.

Recommended result summary so far:

- Logistic Regression baseline: `F1 = 0.7873`
- Linear SVM baseline: `F1 = 0.7695`
- Random Forest baseline: `F1 = 0.7433`
- Fast RoBERTa validation run: `F1 = 0.7932`
- RoBERTa v2 validation run: `F1 = 0.8116`
- RoBERTa v3 validation run: `F1 = 0.8049`
- DeBERTa v3 base validation run: `F1 = 0.8245`
- Best RoBERTa threshold for v2: `0.35`
- Best RoBERTa threshold for v3: `0.420`
- Best DeBERTa threshold: `0.300`

LLM experiment notes:

- Groq prompt-based classification produced a weak validation result and repeatedly hit free-tier request/token limits
- Gemini prompt-based classification hit free-tier request quotas and response issues, so it was not suitable as the main model
- These experiments are worth mentioning in the presentation as an exploration, but they are not the final solution

## Presentation emphasis

The presentation should explain:

- why the problem matters
- how the text was cleaned
- why the chosen models were used
- what the metrics mean
- where the model fails
- whether the dirty labels limit performance
- why RoBERTa was preferred over prompt-based LLM APIs for the final project result

## Implementation status

The notebook should now contain:

1. data loading and inspection
2. conflict detection and cleaning
3. explicit preprocessing steps written out in separate functions
4. baseline classical models with logged F1
5. a fast RoBERTa transformer experiment on a single train/validation split
6. threshold search and submission generation for the transformer run
7. separate Groq and Gemini API experiment notebooks for comparison and limitations

## Current results snapshot

- Logistic Regression baseline: `F1 = 0.7873`
- Linear SVM baseline: `F1 = 0.7695`
- Random Forest baseline: `F1 = 0.7433`
- Fast RoBERTa validation run: `F1 = 0.7932`
- RoBERTa v2 validation run: `F1 = 0.8116`
- Best RoBERTa threshold: `0.35`

The current RoBERTa run is the strongest completed model and is already close to the best public Kaggle range for this competition.

## LLM experiment summary

- Groq notebook: worked technically, but the result was weak and the free-tier limits made full validation unstable
- Gemini notebook: repeatedly hit free-tier quota and response issues, so it was not reliable enough for the final model comparison
- These notebooks remain useful as evidence that the LLM route was explored seriously, but they are not the project's main result

Transformer experiment notes:

- RoBERTa v3 showed that more aggressive tuning does not automatically improve F1 over the softer v2 setup
- DeBERTa v3 base improved the result and is now the best completed transformer model in the project
