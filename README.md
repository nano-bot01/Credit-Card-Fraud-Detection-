# Credit-Card-Fraud-Detection-
Credit Card Fraud Detection using Logistic Regression on credit card dataset 



> As this is a binary classification problem we will be using Logistic Regression model for model training 
### Workflow of model 

* Collection of data
* Data Preprocessing
* Splitting test and training data
* Model Training
* Model Evaluation
* Prediction System


### Dependencies used :
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```


### [Dataset_Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)


```
# importing data

transaction_dataset= pd.read_csv("/content/drive/MyDrive/google_collab/creditcard.csv")
transaction_dataset.head(10)
```


### Data analysis 
  * shape
  * info()
  * describe()
  * isnull
  * count_values()
  * dtypes
 
 
### Sampling 
  - 0 : Normal transaction
  - 1 : Fraudulent transaction

```
legit = transaction_dataset[transaction_dataset.Class == 0]
fraud = transaction_dataset[transaction_dataset.Class == 1]
```

comparing the samples
```
# comparing the values for both transaction 
transaction_dataset.groupby('Class').mean()
```


### Under-Sampling 

- build a sample dataset having similar distribution of normal and fraudulent transactions.
- number of fraudulent transaction is = 492


### Visualization of data 

```
plt.figure(figsize = (20,11))
# heatmap size in ration 16:9

sns.heatmap(new_transaction_dataset2.corr(), annot = True, cmap = 'coolwarm')
# heatmap parameters

plt.title("Heatmap for correlation matrix for credit card data ", fontsize = 22)
plt.show()
```

![001](https://user-images.githubusercontent.com/78251168/221812102-c0a87786-fa08-44a6-b7a5-7b8b26d73e87.png)

![002](https://user-images.githubusercontent.com/78251168/221812120-c05a4036-aaca-4f53-8940-158215258f12.png)


### Splitting data (features and target)
```
X = new_transaction_dataset2.drop(columns = 'Class', axis = 1)
Y = new_transaction_dataset2['Class']
```


### Splitting into training and test
```
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state = 2)
```


### Model Training 
```
model = LogisticRegression()
model.fit(X_train, Y_train)
```


### Model Evaluation 

```
print("\nAccuracy on Training data ",traning_data_accuracy,"\n")
print("Accuracy on Training data ",test_data_accuracy)
```

![image](https://user-images.githubusercontent.com/78251168/221815651-a85aeeaa-863c-4b52-996a-4aee61b61610.png)

<hr>

## Contributor : [Ankit Nainwal](https://github.com/nano-bot01)

### Other Models 
 * [Fake News Prediction System](https://github.com/nano-bot01/Fake-News-Prediction-System-)
 * [Heart Disease Prediction System based on Logistic Regression](https://github.com/nano-bot01/Heart-Disease-Prediction-System-using-Logistic-Regression)

## Please ⭐⭐⭐⭐⭐ 
