## Exercise 1 – Deep Learning for Natural Language and Code

**Prof. Dr. Steffen Herbold — SoSe 2025**

**Due Date: 2025/05/08**

## General Information

In this exercise, you will practice developing your own NLP solutions rather than simply applying existing libraries. A Jupyter Notebook template is provided in StudIP to help you organize your work.

## Problem Overview

You will manipulate text by creating a Bag-of-Words (BoW) representation, build a classifier for sentiment analysis, and implement a text generation model.

Dataset: [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) – 25,000 positive and 25,000 negative movie reviews for training and testing.

## Programming Tasks

### Rules and Tips

- Only standard Python libraries allowed.
- Exceptions: You may use external libraries for:
    - Reading/writing LIBSVM BoW files
    - Porter stemming
    - Random Forest training

### Step-by-Step Tasks

1. **Analyze Provided LIBSVM BoW Files**
    - Investigate the preprocessing steps used.
2. **Build Your Own BoW**
    - Tokenize on spaces and punctuation.
    - Lowercase all text.
    - Remove punctuation.
    - Remove terms appearing in more than a set threshold (e.g., 1%, 5%, 10%).
    - Apply Porter Stemming.
3. **Compare BoWs**
    - Compare your BoW to the one provided and hypothesize reasons for differences.
4. **Train a Classifier**
    - Use your BoW representation to train a Random Forest to classify positive vs negative reviews.
5. **Build a Simple Markov Model**
    - Estimate token probabilities from text, using your pipeline **without** stemming.

## Theoretical Questions

### 1. Categorization of Tasks

Define whether Tasks 2, 4, and 5 fall under NLG, NLU, or NLP.

### 2. Text Processing Pipeline Discussion

Discuss the advantages and drawbacks of steps like stemming and punctuation removal.

### 3. Review Linguistic Concepts

Define:

- Polysemy
- Zeugma
- Homonyms
- Homographs
- Homophones

Then answer:

- Are these concepts important in NLP? Why?
- How do they impact a BoW representation?

### 4. Markov Model Performance

Think about the performance of the Markov model, is it good? Why or why not?

## References

[1] Andrew L. Maas et al., *Learning word vectors for sentiment analysis*, ACL 2011.
