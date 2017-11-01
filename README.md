MLBP Project - Music Genre Classification
==========================================

**Motivation:** Helps group music into categories that can be later used for recommendation or discovery purposes

**Objective:** The idea is to design a complete machine learning model for identifying the music genre of songs

**Problem Statement:** Design a model which learns by reading the train-data (features to labels mapping) and attains the ability to predict labels for the test-data eventually
  - Training: A data set with `4363` songs
  
  - Features: Each song has `264` features `(the first 48 for timbre, the next 48 values represent the pitch and the remaining 168 are for conveying the rhythm patterns)`
  
  - Labels: `10` possible song categories `(1-'Pop_Rock', 2-'Electronic', 3-'Rap', 4-'Jazz', 5-'Latin', 6-'RnB', 7-'International', 8-'Country', 9-'Reggae', 10-'Blues')`
  
  - Testing: A dataset with `6544` songs

**Competition Links:**
  1. [kaggle-accuracy-challenge](https://www.kaggle.com/c/mlbp-2017-da-challenge-accuracy)
  2. [kaggle-logloss-challenge](https://www.kaggle.com/c/mlbp-2017-da-challenge-logloss/rules) #[explanation](https://www.kaggle.com/wiki/LogLoss)
  
**Ideas to try:**
  - Simple logistic regression
  - Simple decsision trees
  - 
  - 
  
**References:**
1. https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/, some obvious classifications so that we can have weights in accordian to the following:

|Mood       |Timbre|Pitch|Rhythm|
|-----------|------|-----|------|
|Happy (reggae, latin)     |Medium|Very High|Very High|
|Exuberant (electronic, rap)  |Medium|High|High|
|Energetic (pop_rock) |Medium|Medium|High|
|Frantic   |Very High|Low|Very High|
|Anxious/Sad (blues)|Very Low|Very Low|Low|
|Depression (blues) |Low|Low|Low|
|Calm (country)      |Very Low|Medium|Very Low|
|Contentment (country)|Low|High|Low|