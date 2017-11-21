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
  
  |  **Logistic Regression** | **modelling strategy** | **solver (learner)** | **weights** | **accuracy** | **loss** |
|  :------: | :------: | :------: | :------: | :------: | :------: |
|   | ovr | lbfgs | None | 0.646 | 0.26 |
|   | multinomial | lbfgs | None | 0.632 | 0.31 |
|   | multinomial | lbfgs | balanced | 0.514 | 0.93 |
|   | ovr | lbfgs | balanced | 0.553 | 0.49 |
|  did not converge after 1200 iterations | ovr | sag/saga | None | 0.658 | 0.12 |
|  did not converge after 1200 iterations | multinomial | sag/saga | None | 0.653 | 0.17 |
|  did not converge after 1200 iterations | multinomial | sag/saga | balanced | 0.529 | 0.59 |
|  did not converge after 1200 iterations | ovr | sag/saga | balanced | 0.564 | 0.39 |
|   | ovr | newton-cg | None | 0.646 | 0.26 |
|   | multinomial | newton-cg | None | 0.631 | 0.39 |
|   | ovr | newton-cg | balanced | 0.553 | 0.49 |
|   | multinomial | newton-cg | balanced | 0.514 | 0.93 |
|   | ovr | liblinear | balanced | 0.587 | 0.37 |
|   | ovr | liblinear | None | 0.65 | 0.21 |

  - Support Vector Classifiers
  
  |  **SVM** | **modelling strategy** | **kernel** | **weights** | **accuracy** | **loss** |
|  :------: | :------: | :------: | :------: | :------: | :------: |
|   | ovo (multinomial) | rbf | None | 0.636 | 0.07 |
|   | ovr | rbf | None | 0.636 | 0.07 |
|   | ovo (multinomial) | linear | None | 0.607 | 0.2 |
|   | ovr | linear | None | 0.607 | 0.2 |
|   | ovr | poly (degree 3) | None | 0.567 | 0.36 |
|   | ovo (multinomial) | poly (degree 3) | None | 0.567 | 0.36 |
|   | ovo (multinomial) | rbf | balanced | 0.579 | 0.08 |
|   | ovr | rbf | balanced | 0.579 | 0.08 |
  
  - ~~Simple decsision trees~~
  - ~~PCA reduction~~ (should we condier other dimensionality reduction strategies?)
  - Random Forests (by calculating appropriate hyper-params)
  - All the above strategies with custom-weights for classes (*training data class frequency:* `(1 - 2178, 2 - 618, 3 - 326, 4 - 253, 5 - 214, 6 - 260, 7 - 141, 8 - 195, 9 - 92, 10 - 86)`)
  - Ensemble of logistic regression and SVM
  
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
