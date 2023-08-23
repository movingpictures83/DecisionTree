#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model
np.random.seed(1234)

# In[328]:


################################################ Preprocessing #########################################################
class DecisionTreePlugin:
 def input(self, inputfile):
  self.data_path = inputfile

 def run(self):
  pass

 def output(self, outputfile):
  #categorical_cols = ["Race_Ethnicity"]


  data_df = pd.read_csv(self.data_path)

  # # Tramsform categorical data to categorical format:
  # for category in categorical_cols:
  #     data_df[category] = data_df[category].astype('category')
  #

  # Clean numbers:
  #"Cocain_Use": {"yes":1, "no":0},
  cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
  }

  data_df.replace(cleanup_nums, inplace=True)

  # Drop id column:
  data_df = data_df.drop(["pilotpid"], axis=1)

  # remove NaN:
  data_df = data_df.fillna(0)

  # Standartize variables
  from sklearn import preprocessing
  names = data_df.columns
  scaler = preprocessing.StandardScaler()
  data_df_scaled = scaler.fit_transform(data_df)
  data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)


  # In[303]:


  ################################################ Users vs Non-Users #########################################################

  # Random Forest
  # Benchmark

  y_col = "Cocain_Use"
  test_size = 0.3
  validate = True


  y = data_df[y_col]

  X = data_df_scaled.drop([y_col], axis=1)

  # Create random variable for benchmarking
  X["random"] = np.random.random(size= len(X))

  X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = test_size, random_state = 2)

  rf = RandomForestClassifier(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 42)

  rf.fit(X_train, y_train)

  print('Training accuracy: {:.2f} \nOOB Score: {:.2f} \nTest Accuracy: {:.2f}'.format(rf.score(X_train, y_train),
                                                                                             rf.oob_score_,
                                                                                             rf.score(X_valid, y_valid)))
  # scores = cross_val_score(rf, X, y, cv=5)
  # print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
  # a = rf.predict(X_valid)

  # importances_df = drop_col_feat_imp(rf, X, y, X_valid, y_valid)
  # importances_df.to_csv("/Users/stebliankin/Desktop/SabrinaProject/FeatureSelection/importance_df.csv")

  scores = cross_val_score(rf, X, y, cv=5)
  print("CV Accuracy: %0.2f " % (scores.mean()))


  # In[122]:


  data_df_scaled.head()


  # In[224]:


  validate=True

  from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
  #from sklearn.model_selection import train_test_split # Import train_test_split function
  from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


  # Create Decision Tree classifer object
  clf = DecisionTreeClassifier()

  # Train Decision Tree Classifer
  clf = clf.fit(X_train,y_train)

  y_pred = clf.predict(X_train)
  print("Train accuracy:",metrics.accuracy_score(y_train, y_pred))

  y_pred = clf.predict(X_valid)
  print("Test accuracy:",metrics.accuracy_score(y_valid, y_pred))


  if validate:
    scores = cross_val_score(rf, X, y, cv=5)
    print("CV Accuracy: %0.2f " % (scores.mean()))


  # In[ ]:




