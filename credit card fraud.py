# IMPORTING PACKAGES

import pandas as pd                     # data processing
import numpy as np                      # working with arrays
import matplotlib.pyplot as plt         # visualization
from termcolor import colored as cl     # text customization
import itertools                        # advanced tools

from sklearn.preprocessing import StandardScaler        # data normalization
from sklearn.model_selection import train_test_split    # data split
from sklearn.tree import DecisionTreeClassifier         # decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier      # KNN algorithm
from sklearn.linear_model import LogisticRegression     # Logistic Regression algorithm
from sklearn.svm import SVC                             # SVM algorithm
from sklearn.ensemble import RandomForestClassifier     # Random forest Tree algorithm
from xgboost import XGBClassifier                       # XGBoost algorithm

from sklearn.metrics import confusion_matrix            # Evaluation metric
from sklearn.metrics import accuracy_score              # Evaluation metric
from sklearn.metrics import f1_score                    # Evaluation metric

# IMPORT DATA

df = pd.read_csv('credit card.csv')
df.drop('Time', axis=1, inplace=True)

print(df.head())

# DATA PROCESSING and EDA

cases = len(df)
total_count = len(df[df.Class == 1] + df[df.Class == 0])
fraud_count = len(df[df.Class == 1])
non_fraud_count = len(df[df.Class == 0])
fraud_percentage = round(((fraud_count/(fraud_count+non_fraud_count))*100), 2)

print(cl('CASE COUNTS', attrs=['bold']))
print(cl('----------------------------------', attrs=['bold']))
print(cl(f'Total number of cases are {total_count}', attrs=['bold']))
print(cl(f'Number of non fraud cases are {non_fraud_count}', attrs=['bold']))
print(cl(f'Number of fraud cases are {fraud_count}', attrs=['bold']))
print(cl(f'The fraud percentage is {fraud_percentage}%', attrs=['bold']))
print(cl('----------------------------------', attrs=['bold']))

# STATISTICAL VIEW

fraud_cases = df[df.Class == 1]
non_fraud_cases = df[df.Class == 0]

print(cl('CASE AMOUNT STATISTICS', attrs=['bold']))
print(cl('----------------------------------', attrs=['bold']))
print(cl('NON FRAUD CASE AMOUNT STATISTICS', attrs=['bold']))
print(non_fraud_cases.Amount.describe())
print(cl('----------------------------------', attrs=['bold']))
print(cl('FRAUD CASE AMOUNT STATISTICS', attrs=['bold']))
print(fraud_cases.Amount.describe())
print(cl('----------------------------------', attrs=['bold']))

# VARIABLE 'AMOUNT' NORMALIZATION

sc = StandardScaler()
amount = df['Amount'].values

df['Amount'] = sc.fit_transform(amount.reshape(-1, 1))

print(cl(df['Amount'].head(10), attrs=['bold']))

# FEATURE SELECTION AND DATA SPLIT

X = df.drop('Class', axis=1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(cl('X_train samples : ', attrs=['bold']), X_train[:1])
print(cl('X_test samples : ', attrs=['bold']), X_test[0:1])
print(cl('y_train samples : ', attrs=['bold']), y_train[0:20])
print(cl('y_test samples : ', attrs=['bold']), y_test[0:20])

# MODELING

# 1. Decision Tree

tree_model = DecisionTreeClassifier(max_depth=4, criterion='entropy')
tree_model.fit(X_train, y_train)
tree_yhat = tree_model.predict(X_test)

# 2. K-Nearest Neighbors

n = 5

knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X_train, y_train)
knn_yhat = knn.predict(X_test)

# 3. Logistic Regression

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

# 4. SVM

svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)

# 5. SVM

rf = RandomForestClassifier(max_depth=4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)

xgb = XGBClassifier(max_depth=4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

# 1. Accuracy score

print(cl('ACCURACY SCORE', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'Accuracy score of the Decision Tree model is {accuracy_score(y_test, tree_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'Accuracy score of the KNN model is {accuracy_score(y_test, knn_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'Accuracy score of the Logistic Regression model is {accuracy_score(y_test, lr_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'Accuracy score of the SVM model is {accuracy_score(y_test, svm_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'Accuracy score of the Random Forest Tree model is {accuracy_score(y_test, rf_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'Accuracy score of the XGBoost model is {accuracy_score(y_test, xgb_yhat)}', attrs=['bold']))

# 1. F1 score

print(cl('F1 SCORE', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'F1 score of the Decision Tree model is {f1_score(y_test, tree_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'F1 score of the KNN model is {f1_score(y_test, knn_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'F1 score of the Logistic Regression model is {f1_score(y_test, lr_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'F1 score of the SVM Tree model is {f1_score(y_test, svm_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'F1 score of the Random Forest Tree model is {f1_score(y_test, rf_yhat)}', attrs=['bold']))
print(cl('------------------------------------------------------------------------', attrs=['bold']))
print(cl(f'F1 score of the XGBoost model is {f1_score(y_test, xgb_yhat)}', attrs=['bold']))

# 3. Confusion Matrix

# defining the plot function


def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    title = f'Confusion Matrix of {title}'
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# Compute confusion matrix for the models


tree_matrix = confusion_matrix(y_test, tree_yhat, labels=[0, 1])        # Decision Tree
knn_matrix = confusion_matrix(y_test, knn_yhat, labels=[0, 1])          # K-Nearest Neighbors
lr_matrix = confusion_matrix(y_test, lr_yhat, labels=[0, 1])            # Logistic Regression
svm_matrix = confusion_matrix(y_test, svm_yhat, labels=[0, 1])          # Support Vector Machine
rf_matrix = confusion_matrix(y_test, rf_yhat, labels=[0, 1])            # Random Forest Tree
xgb_matrix = confusion_matrix(y_test, xgb_yhat, labels=[0, 1])          # XGBoost

plt.rcParams['figure.figsize'] = (6, 6)

# 1. Decision Tree

tree_cm_plot = plot_confusion_matrix(tree_matrix,
                                     classes=['Non-Default(0)', 'Default(1)'],
                                     normalize=False, title='Decision Tree')
plt.savefig('tree_cm_plot.png')
plt.show()

# 2. K-Nearest Neighbors

knn_cm_plot = plot_confusion_matrix(knn_matrix,
                                     classes=['Non-Default(0)', 'Default(1)'],
                                     normalize=False, title='KNN')
plt.savefig('knn_cm_plot.png')
plt.show()

# 3. Logistic regression

lr_cm_plot = plot_confusion_matrix(lr_matrix,
                                classes = ['Non-Default(0)','Default(1)'],
                                normalize = False, title = 'Logistic Regression')
plt.savefig('lr_cm_plot.png')
plt.show()

# 4. Support Vector Machine

svm_cm_plot = plot_confusion_matrix(svm_matrix,
                                classes = ['Non-Default(0)','Default(1)'],
                                normalize = False, title = 'SVM')
plt.savefig('svm_cm_plot.png')
plt.show()

# 5. Random forest tree

rf_cm_plot = plot_confusion_matrix(rf_matrix,
                                classes = ['Non-Default(0)','Default(1)'],
                                normalize = False, title = 'Random Forest Tree')
plt.savefig('rf_cm_plot.png')
plt.show()

# 6. XGBoost

xgb_cm_plot = plot_confusion_matrix(xgb_matrix,
                                classes = ['Non-Default(0)','Default(1)'],
                                normalize = False, title = 'XGBoost')
plt.savefig('xgb_cm_plot.png')
plt.show()
