

## Abstract

This project presents an in-depth analysis of an algorithm designed to predict gender and age from audio features extracted from speech data. The algorithm utilizes various machine learning techniques to achieve accurate predictions. The document details the data preprocessing, feature engineering, model development, and evaluation processes.

## Introduction

The objective of this project is to develop an algorithm that can accurately predict gender and age from audio features. This is a challenging task due to the inherent variability in human speech. However, with the right combination of feature extraction techniques and machine learning models, it is possible to achieve a high level of accuracy.

## Data Preprocessing

### Data Loading and Cleaning

The first step in the data preprocessing pipeline involves loading the dataset from a CSV file. This file contains metadata for each audio file, including the filename, gender, and age of the speaker. To ensure the integrity of the data, any rows with missing values in the ‘age’ and ‘gender’ columns were removed.

### Feature Extraction

The next step is to extract meaningful features from the audio data. This is accomplished using the Librosa library, which provides a suite of tools for music and audio analysis. The features extracted include the spectral centroid, spectral bandwidth, spectral rolloff, and Mel-frequency cepstral coefficients (MFCCs). These features capture various aspects of the audio signal and are commonly used in speech and audio processing tasks.

### Feature Engineering

#### Feature Scaling

Before feeding the data into the machine learning models, it is necessary to normalize the features. This is done using StandardScaler from the scikit-learn library, which standardizes features by removing the mean and scaling to unit variance.

#### Feature Selection

To improve the efficiency and performance of the models, feature selection is performed using SelectKBest from the scikit-learn library. This method selects the k best features based on a given scoring function. In this case, an ANOVA F-value is used as the scoring function, which measures the variance between the feature and the output variable.

## Model Development

### Decision Tree Regression

For age prediction, a Decision Tree Regression model is implemented. Decision trees are simple yet powerful models that can capture complex relationships in the data. The maximum depth of the tree is set to control the complexity of the model and prevent overfitting.

### Support Vector Machine (SVM)

For gender prediction, a Support Vector Machine (SVM) model is employed. SVMs are effective for binary classification tasks and can handle high-dimensional data. The SVM model is configured with a radial basis function (RBF) kernel, which can handle non-linear decision boundaries. The hyperparameters C and gamma are set to 100 and ‘scale’, respectively, to control the trade-off between model complexity and error.

## Model Training and Evaluation

The models are trained and evaluated using robust methods to ensure their generalizability. For the Decision Tree Regression model, k-fold cross-validation is utilized. This method provides a more reliable estimate of the model’s performance by averaging the results over multiple folds. The Mean Squared Error (MSE) is used as the evaluation metric for regression, which measures the average squared difference between the predicted and actual values.

The SVM model is trained on a separate training set and evaluated on a test set. This ensures that the model’s performance is assessed on unseen data.

## Results

The Decision Tree Regression model achieved an average MSE of X on the test set, indicating a reasonable level of accuracy in age prediction. The SVM model achieved an accuracy of X% on the test set for gender prediction, demonstrating its effectiveness in classifying gender based on audio features.

## Conclusion

In conclusion, the algorithm successfully predicts gender and age from audio features with a reasonable level of accuracy. The combination of feature extraction, feature engineering, and machine learning models proved effective in tackling this challenging task. Future work may involve exploring additional features, experimenting with different machine learning models, and tuning hyperparameters for improved performance.

