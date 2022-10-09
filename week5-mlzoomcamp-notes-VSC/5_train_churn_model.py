#!/usr/bin/env python
# coding: utf-8

# In the previous session we trained a model for predicting churn and evaluated it. Now let's deploy it

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[2]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[5]:


df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[6]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[7]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[8]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[9]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[10]:


C = 1.0
n_splits = 5


# In[11]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[12]:


scores


# In[13]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# Save the model

# In[14]:


import pickle


# In[15]:


output_file = f'model_C={C}.bin'
output_file


# In[16]:


f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()


# In[17]:


get_ipython().system('ls -lh *.bin')


# In[18]:


with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)


# Load the model
# 
# 

# In[1]:


import pickle


# In[2]:


input_file = 'model_C=1.0.bin'


# In[3]:


with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)


# In[4]:


model


# In[5]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[6]:


X = dv.transform([customer])


# In[7]:


y_pred = model.predict_proba(X)[0, 1]


# In[8]:


print('input:', customer)
print('output:', y_pred)


# Making requests

# In[8]:


import requests


# In[9]:


url = 'http://localhost:9696/predict'


# In[10]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'two_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[11]:


response = requests.post(url,json=customer).json()


# In[ ]:


response


# In[ ]:


if response['churn']:
    print('sending email to', 'asdx-123d')


#  create a simple web service

# In[ ]:


from flask import Flask
app = Flask('churn-app') 
@app.route('/ping',methods=[GET])
def ping():
    return 'PONG'

if __name__=='__main__':
   app.run('debug=True, host='0.0.0.0', port=9696) 


# In[ ]:


import pickle

with open('churn-model.bin', 'rb') as f_in:
  dv, model = pickle.load(f_in)


# In[2]:


def predict_single(customer, dv, model):
  X = dv.transform([customer])  
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]


# In[ ]:


@app.route('/predict', methods=['POST'])  
def predict():
  customer = request.get_json()
  
  prediction = predict_single(customer, dv, model)
  churn = prediction >= 0.5

result = {
    'churn_probability': float(prediction), 
    'churn': bool(churn),  
}

return jsonify(result)  


# In[ ]:


## a new customer informations
customer = {
  'customerid': '8879-zkjof',
  'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'no',
  'dependents': 'no',
  'tenure': 41,
  'phoneservice': 'yes',
  'multiplelines': 'no',
  'internetservice': 'dsl',
  'onlinesecurity': 'yes',
  'onlinebackup': 'no',
  'deviceprotection': 'yes',
  'techsupport': 'yes',
  'streamingtv': 'yes',
  'streamingmovies': 'yes',
  'contract': 'one_year',
  'paperlessbilling': 'yes',
  'paymentmethod': 'bank_transfer_(automatic)',
  'monthlycharges': 79.85,
  'totalcharges': 3320.75
}
import requests
url = 'http://localhost:9696/predict' 
response = requests.post(url,json=customer) 
result = response.json() 
print(result)


# In[ ]:


from flask import Flask
app = Flask('churn-app') 
@app.route('/ping',methods=[GET])
def ping():
    return 'PONG'

if __name__=='__main__':
   app.run('debug=True, host='0.0.0.0', port=9696')


# In[ ]:


import pickle

with open('churn-model.bin', 'rb') as f_in:
  dv, model = pickle.load(f_in)


# In[ ]:


def predict_single(customer, dv, model):
  X = dv.transform([customer])  
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]


# In[ ]:


@app.route('/predict', methods=['POST'])  
def predict():
customer = request.get_json()  

prediction = predict_single(customer, dv, model)
churn = prediction >= 0.5

result = {
    'churn_probability': float(prediction), 
    'churn': bool(churn),  
}

return jsonify(result)  


# In[ ]:


## a new customer informations
customer = {
  'customerid': '8879-zkjof',
  'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'no',
  'dependents': 'no',
  'tenure': 41,
  'phoneservice': 'yes',
  'multiplelines': 'no',
  'internetservice': 'dsl',
  'onlinesecurity': 'yes',
  'onlinebackup': 'no',
  'deviceprotection': 'yes',
  'techsupport': 'yes',
  'streamingtv': 'yes',
  'streamingmovies': 'yes',
  'contract': 'one_year',
  'paperlessbilling': 'yes',
  'paymentmethod': 'bank_transfer_(automatic)',
  'monthlycharges': 79.85,
  'totalcharges': 3320.75
}
import requests 
url = 'http://localhost:9696/predict' 
response = requests.post(url, json=customer) 
result = response.json() 
print(result)


# In[ ]:


bash
sudo apt-get install docker.io


# In[ ]:


FROM python:3.8.12-slim                                                     
RUN pip install pipenv                                                      
WORKDIR /app                                                                
COPY ["Pipfile", "Pipfile.lock", "./"]                                      
RUN pipenv install --deploy --system                                        
COPY ["*.py", "churn-model.bin", "./"]                                      
EXPOSE 9696                                                                 
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "churn_serving:app"]      


# In[ ]:


docker build -t churn-prediction .


# In[ ]:


docker run -it -p 9696:9696 churn-prediction:latest

