#Maybe introduce bad data to the dataset

#Hyperparameters are parameters set before the learning begins to control how a model learns.
#These are for data manipulation and model building
import pandas as pd #handles dattaframes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

#libraries for ml model selecetion and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV #Splits data and performs hyperparameter opitimisation

#ML models
from sklearn.svm import SVC #support vector classifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

#model performance evaluation
from sklearn.metrics import accuracy_score


#Load and preprocess the dataset.
dataset = pd.read_csv('Student_performance_data _.csv')

#Removes any empty and replaces them with average.
dataset.drop(["StudentID"], axis=1, inplace=True)#Student ID is removed as it is irrelivant and GPA is removed as it is too similar to grade class (Target variable).


#Encodes catagorical variables
#dataset['Gender'] = dataset['Gender'].map({'male': 0, 'female': 1})
#dataset['Ethnicity'] = dataset['Ethnicity'].map({'Caucasian': 0, 'African American': 1, 'Asian': 2, 'Other': 3})
#dataset['ParentalEducation'] = dataset['ParentalEducation'].map({'None' : 0, 'High School': 1, 'Some College': 2, 'Bachelor\'s': 3, 'Higher': 4})
#dataset['ParentalSupport'] = dataset['ParentalSupport'].map({'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4})


#Filling in missing values
dataset['Age'].fillna(dataset['Age'].median(), inplace=True)#Replaces any empty coloumns of age with the median
dataset['StudyTimeWeekly'].fillna(dataset['StudyTimeWeekly'].median(), inplace=True)#Replaces any empty coloumns of Study time weekly with the median
dataset['Absences'].fillna(dataset['Absences'].median(), inplace=True)#Replaces any empty coloumns of Absences with the median

dataset['Gender'].fillna(dataset['Gender'].mode(), inplace=True)
dataset['Ethnicity'].fillna(dataset['Ethnicity'].mode(), inplace=True)
dataset['ParentalEducation'].fillna(dataset['ParentalEducation'].mode(), inplace=True)
dataset['Tutoring'].fillna(dataset['Tutoring'].mode(), inplace=True)
dataset['ParentalSupport'].fillna(dataset['ParentalEducation'].mode(), inplace=True)
dataset['Extracurricular'].fillna(dataset['Extracurricular'].mode(), inplace=True)
dataset['Sports'].fillna(dataset['Sports'].mode(), inplace=True)
dataset['Music'].fillna(dataset['Music'].mode(), inplace= True)
dataset['Volunteering'].fillna(dataset['Volunteering'].mode(), inplace=True)


X = dataset.drop('GradeClass', axis=1) #Everything but target variable
Y = dataset['GradeClass']#This contains the target variable

#Synthetic Minority Over - sampling
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

#Splitting the dataset into training and test.
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=45)

#Creating dictionarary to store models and hyperparameters to optimise.
model_params = {
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [20, 50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    'xgboost': {
        'model': XGBClassifier(eval_metric='logloss'),
        'params': {
            'n_estimators': [25, 50, 100],
            'learning_rate': [1, 0.1, 0.01],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1]
        }
    }
}

#performing hyperparameter optimisation

#stores results of GridSearchCV
results = []#Initialises an empty list to store model results

#Iterates through each model and its parameters
for model_name, mp in model_params.items():
    print(f"Training {model_name}...")#Says which model is being trained
    
    #GridSearchCV with model, parameteres, and cross-validation settings
    grid_search = GridSearchCV(mp['model'], mp['params'], cv=5, scoring = 'accuracy') #5-fold cross-validation
    #Fits the GridSearchCV instance on the training data
    grid_search.fit(X_train, Y_train) #Trains the model on the training data
    #Append model results to the results list

    Best_estimator = grid_search.best_estimator_

    #Feature importance
    feature_importance = None
    if hasattr(Best_estimator, 'feature_importances_'):
        feature_importance = Best_estimator.feature_importances_

    results.append({
        'model': model_name, #Name of the model
        'best_params': grid_search.best_params_, #Best hyperparameters found
        'best_score': grid_search.best_score_, #Best crost-validation score
        'best_estimator': Best_estimator,
        'prediction' : grid_search.predict(X_test),
        'optimised_prediction' : Best_estimator.predict(X_test),
        'feature_names' : grid_search.feature_names_in_,
        'feature_importance' : feature_importance
    })


#Evaluating the best models

Prediction_randomTree_Op = results[0]['optimised_prediction']
accuracy_randomTree_Op = accuracy_score(Y_test, Prediction_randomTree_Op)

Prediction_Xgboost_Op = results[1]['optimised_prediction']
accuracy_Xgboost_Op = accuracy_score(Y_test, Prediction_Xgboost_Op)

print(f"Optimised Accuracy Scores\n Random tree:\n {accuracy_randomTree_Op}\n Xgboost:\n {accuracy_Xgboost_Op}")

#Ensemble learning uses multiple models to make predictions more accurate.

ensemble_model = VotingClassifier(
    estimators=[
        ('random_forest', results[0]["best_estimator"]),
        ('xgboost', results[1]["best_estimator"]),
    ],
    voting='hard'
)

ensemble_model.fit(X_train, Y_train) #Training the combined model
Ensemble_Prediction = ensemble_model.predict(X_test)
Ensemble_Accuracy = accuracy_score(Y_test, Ensemble_Prediction)
print(f"Ensemble Accuracy:\n {Ensemble_Accuracy}")

models_results = {
    "Random_Forest" : (results[0]["best_estimator"], Prediction_randomTree_Op),
    "XgBoost" : (results[1]["best_estimator"], Prediction_Xgboost_Op),
    "Ensemble" : (ensemble_model, Ensemble_Prediction)
}


#Visualisation of the models

#This displays a heatmap showing the amount of correctley predicted outputs
for model_name, (model, Y_pred) in models_results.items():
    # Confusion matrix
    matrix = confusion_matrix(Y_test, Y_pred)
    #Prints out the classification report
    print(f"Classification Report for {model_name}:\n{classification_report(Y_test, Y_pred)}")


    print(f"{model_name}'s Confusion Matrix:\n{matrix}")
   
    # Calculate and display details for each class
    for i in range(len(matrix)):  # Loop through each class (diagonal entries)
        true_positive = matrix[i, i]  # Diagonal entry (correct predictions for class i)
        total_samples_in_class = matrix[i, :].sum()  # Total samples belonging to class i
        print(
            f"Class {i}: {true_positive} samples were correctly predicted as {i} "
            f"out of {total_samples_in_class} total samples."
        )
   
    # Visualize the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, cmap="Greens", xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'], fmt="d")
    plt.title(f"{model_name}'s Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


#This displays a heatmap showing the accuracy for each class
for model_name, (model, Y_pred) in models_results.items():
    # Confusion matrix
    matrix = confusion_matrix(Y_test, Y_pred)
    print(f"{model_name}'s Confusion Matrix:\n{matrix}")
 
    # Normalized confusion matrix (row-wise normalization)
    normalized_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
 
    # Visualize the normalized confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        normalized_matrix, annot=True, cmap="Blues", xticklabels=['0', '1', '2', '3', '4'], yticklabels=['0', '1', '2', '3', '4'], fmt=".2f"
    )
    plt.title(f"{model_name}'s Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
 
    print(f"{model_name}'s Normalized Confusion Matrix (row-wise):\n{normalized_matrix}")

#Displays feature importance of random forest.
importances = results[1]['feature_importance']
forest_importances = pd.Series(importances, index = results[1]["feature_names"])
plt.figure(figsize=(10, 6))
plt.title("Feature Importances For Random Forest")
plt.bar(forest_importances.index, forest_importances.values)
plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()