import pandas as pd
import numpy as np

#pip install scikit-learn
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

df = pd.read_csv('Creditcard_data.csv')
df.head()

#pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x = 'Class',data = df,color='purple')
plt.show()        

#pip install imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SVMSMOTE, SMOTE, BorderlineSMOTE, ADASYN


features = df.drop("Class", axis=1)
target = df["Class"]

random_state_value = 45

# Random Under Sampler
rus = RandomUnderSampler(random_state=random_state_value)
features_rus, target_rus = rus.fit_resample(features, target)
df_rus = pd.concat([features_rus, target_rus], axis=1)
df_rus.to_csv("data_rus.csv", index=False)

# SVMSMOTE
sms = SVMSMOTE(random_state=random_state_value)
features_sms, target_sms = sms.fit_resample(features, target)
df_sms = pd.concat([features_sms, target_sms], axis=1)
df_sms.to_csv("data_sms.csv", index=False)

# SMOTE
smote = SMOTE(random_state=random_state_value)
features_smote, target_smote = smote.fit_resample(features, target)
df_smote = pd.concat([features_smote, target_smote], axis=1)
df_smote.to_csv("data_smote.csv", index=False)

# Border Line SMOTE
bs = BorderlineSMOTE(random_state=random_state_value)
features_bs, target_bs = bs.fit_resample(features, target)
df_bs = pd.concat([features_bs, target_bs], axis=1)
df_bs.to_csv("data_bs.csv", index=False)

# ADASYN Over-sampling
ad = ADASYN(random_state=random_state_value)
features_ad, target_ad = ad.fit_resample(features, target)
df_ad = pd.concat([features_ad, target_ad], axis=1)
df_ad.to_csv("data_ad.csv", index=False)

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = ["data_rus.csv",
            "data_sms.csv",
            "data_smote.csv",
            "data_bs.csv",
            "data_ad.csv"]

models = [LogisticRegression(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          GaussianNB(),
          GradientBoostingClassifier()]

sampling_techniques = ['RandomUnderSampler', 'SVMSMOTE', 'SMOTE', 'BorderlineSMOTE', 'ADASYN']

results = []

for dataset, sampling_technique in zip(datasets, sampling_techniques):
    try:
        df = pd.read_csv(dataset)
    except FileNotFoundError:
        print(f"Error: File {dataset} not found. Check the file path.")
        continue

    X = df.drop("Class", axis=1)
    y = df["Class"]

    for model, model_name in zip(models, ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GaussianNB' , 'GradientBoostingClassifier']):
        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model and make predictions
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        # Calculate accuracy and store results
        accuracy = accuracy_score(Y_test, y_pred)
        results.append({'Sampling': sampling_technique, 'Classifier': model_name, 'Accuracy': 100 * accuracy})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
results_df.head()

pivot_df = results_df.pivot_table(index='Classifier', columns='Sampling', values='Accuracy')
pivot_df.to_csv('final.csv')

pivot_df.head()
