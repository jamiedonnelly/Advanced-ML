import GPy; import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; 
from sklearn.model_selection import train_test_split; import seaborn as sns; from sklearn.metrics import accuracy_score;
from sklear.metrics import confusion_matrix; from sklearn.metrics import mean_squared_error;
from sklearn.model_selection import KFold; from sklearn.decomposition import PCA;


data = pd.read_csv("/Users/jamie/Desktop/weather.csv")

data.head(10)

plt.figure(figsize=(30,10))
plt.plot(data['Temperature (C)'])

data = data[:200]

plt.figure(figsize=(16,8))
plt.plot(data['Temperature (C)'])
plt.title("Temperature")
plt.ylabel("Temp (C)")
plt.xlabel("Time")

data.columns
X_df = data.drop(['Formatted Date','Summary','Precip Type','Temperature (C)','Apparent Temperature (C)','Daily Summary'],axis=1)
X_df.head(10)

y_df = data['Temperature (C)']
y_df.head()

n = len(data)
X = np.array(X_df).reshape(n,6)
y = np.array(y_df).reshape(n,1)

Xtr, Xts, Ytr, Yts = train_test_split(X,y,test_size=0.2)

variances = [0.1,0.2,0.4,1,2,4,8]
lengthscales = [0.1,0.2,0.4,1,2,4,8]
d = 6
results = pd.DataFrame(columns=variances,index=lengthscales)
for i in variances:
    for j in lengthscales: 
        k = GPy.kern.RBF(d,i,j)
        m = GPy.models.GPRegression(Xtr,Ytr,k)
        preds = np.array(m.predict(Xts))[0]
        results.loc[j,i] = mean_squared_error(Yts,preds)

results

k = GPy.kern.RBF(6,8,8)

# Sampling from prior with chosen cov mat and mean vector of 0s
m = np.array([0]*len(Xtr[:50]))
C = k.K(Xtr[:50],Xtr[:50])

plt.figure(figsize=(20,8))
plt.ylim(-10,10)
plt.plot(np.linspace(-5,5,len(Xtr[:50])),np.random.multivariate_normal(m,C))
plt.plot(np.linspace(-5,5,len(Xtr[:50])),np.random.multivariate_normal(m,C))
plt.plot(np.linspace(-5,5,len(Xtr[:50])),np.random.multivariate_normal(m,C))
plt.title("Sample functions from the prior distribution")
plt.legend(['Sample 1','Sample 2','Sample 3'])

# Since I am using a non-sklearn model and using GPy I will need to
# manually apply a 10-fold CV to assess performance 

k_fold = KFold(10)
scores =[]
k = GPy.kern.RBF(6,8,8)
for i, (train, test) in enumerate(k_fold.split(X, y)):
    m = GPy.models.GPRegression(X[train],y[train],k)
    preds = np.array(m.predict(X[test]))[0]
    scores.append(mean_squared_error(y[test],preds))
    plt.scatter(y[test],preds)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend([i for i in range(1,11)])

scores
np.mean(scores)
np.std(scores)


# Timing the cross-validation regression function 

start = time()
k_fold = KFold(10)
scores =[]
k = GPy.kern.RBF(6,8,8)
for i, (train, test) in enumerate(k_fold.split(X, y)):
    m = GPy.models.GPRegression(X[train],y[train],k)
    preds = np.array(m.predict(X[test]))[0]
    scores.append(mean_squared_error(y[test],preds))
end = time()

print("Time taken:",end-start,"seconds")
# 0.1245 seconds 

##### CLASSIFICATION ######

#Observe distribution 

# Min/Max and median values
min(y_df), max(y_df)
med = np.median(y_df)

plt.figure(figsize=(12,8))
x = sns.distplot(y_df,bins=25)
plt.title("Histogram of Temperatures")
plt.vlines(med,ymin=0,ymax=0.15)

# Encode data 

for i in range(len(y_df)):
    if min(y_df) <= y_df[i] <= median:
        y_df[i] = 0
    elif y_df[i] > median:
        y_df[i] = 1

# Check for value counts 
y_df.value_counts()

# Get into array form for the regression problem 
X = np.array(X_df).reshape(200,6)
y = np.array(y_df).reshape(200,1)
Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=0.2)

# Parameter Tuning for the var and lengthscale

variances = [0.1,0.2,0.4,1,2,4,8]
lengthscales = [0.1,0.2,0.4,1,2,4,8]
d=2
results = pd.DataFrame(columns=variances,index=lengthscales)
for i in variances:
    for j in lengthscales: 
        k = GPy.kern.RBF(d,i,j)
        m = GPy.models.GPClassification(Xtr,ytr,k)
        preds = np.array(m.predict(Xts))[0]
        preds[preds <= 0.5] = 0
        preds[preds > 0.5] = 1
        results.loc[j,i] = accuracy_score(yts,preds)

results

# Determine best threshold
for i in np.linspace(0,1,20):
    k = GPy.kern.RBF(2,8,2)
    m = GPy.models.GPClassification(Xtr,ytr,k)
    m.optimize()
    preds = np.array(m.predict(Xts))[0]
    preds[preds > i] = 1 
    preds[preds <= i] = 0
    plt.scatter(i,accuracy_score(yts,preds),color='red')
    plt.vlines(0.50,ymin=0,ymax=1)
    plt.title("Accuracy as a function of threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Test Accuracy")


# Cross-val for classification 

k_fold = KFold(10)
scores =[]
k = GPy.kern.RBF(2,8,2)
for i, (train, test) in enumerate(k_fold.split(X, y)):
    m = GPy.models.GPClassification(X[train],y[train],k)
    preds = m.predict(X[test])[0]
    preds[preds > 0.5] = 1 
    preds[preds <= 0.5] = 0
    scores.append(accuracy_score(y[test],preds))

print(scores, np.mean(scores), np.std(scores))

# Timing of the classification process 

start = time()
k_fold = KFold(10)
scores =[]
k = GPy.kern.RBF(2,8,2)
for i, (train, test) in enumerate(k_fold.split(X, y)):
    m = GPy.models.GPClassification(X[train],y[train],k)
    preds = m.predict(X[test])[0]
    preds[preds > 0.5] = 1 
    preds[preds <= 0.5] = 0
    scores.append(accuracy_score(y[test],preds))
end = time()

print("Time taken:",end-start,"seconds")
# 1.683 seconds 

## PCA and Kernel Functions

model = PCA(n_components=1)
xx = model.fit_transform(X)
print(model.explained_variance_ratio_)

plt.figure(figsize=(8,8))
plt.scatter(model.fit_transform(xx),y)
plt.title("First Principal Component vs Temperature")
plt.xlabel("Z_1")
plt.ylabel("Temperature (C)")


GPy.kern.RatQuad(1,1,1).plot()
GPy.kern.RatQuad(1,8,8).plot()
GPy.kern.Exponential(1,1,1).plot()
GPy.kern.Exponential(1,8,8).plot()


k = ... # Insert variety of kernels here for regression
k_fold = KFold(10)
scores =[]
for i, (train, test) in enumerate(k_fold.split(xx, y)):
    m = GPy.models.GPRegression(xx[train],y[train],k)
    preds = np.array(m.predict(xx[test]))[0]
    scores.append(mean_squared_error(y[test],preds))
    plt.scatter(y[test],preds)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend([i for i in range(1,11)])
np.mean(scores)





