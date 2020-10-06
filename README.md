
# Bias-Variance Tradeoff


## Agenda

1. Revisit the goal of model building, and relate it to expected value, bias and variance
2. Defining Error: prediction error and irreducible error
3. Define prediction error as a combination of bias and variance
4. Explore the bias-variance tradeoff
5. Code a basic train-test split
6. Code K-Folds



# 1. Revisit the goal of model building, and relate it to expected value, bias and variance

![which model is better](img/which_model_is_better.png)

https://towardsdatascience.com/cultural-overfitting-and-underfitting-or-why-the-netflix-culture-wont-work-in-your-company-af2a62e41288


# What makes a model good?

- We don’t ultimately care about how well your model fits your data.

- What we really care about is how well your model describes the process that generated your data.

- Why? Because the data set you have is but one sample from a universe of possible data sets, and you want a model that would work for any data set from that universe

# What is a “Model”?

 - A “model” is a general specification of relationships among variables. 
     - E.G. Linear Regression: or $ Price = \beta_1*Y_{t-1} +  \beta_0 + \epsilon$


 

 - A “trained model” is a particular model with parameters estimated using some training data.

# Remember Expected Value? How is it connected to bias and variance?
- The expected value of a quantity is the weighted average of that quantity across all possible samples

![6 sided die](https://media.giphy.com/media/sRJdpUSr7W0AiQ3RcM/giphy.gif)

- for a 6 sided die, another way to think about the expected value is the arithmetic mean of the rolls of a very large number of independent samples.  

### The expected value of a 6-sided die is:


```python
probs = 1/6
rolls = range(1,7)

expected_value = sum([probs * roll for roll in rolls])
expected_value
```

Now lets imagine we create a model that always predicts a roll of 3.

   
  - The bias is the difference between the average prediction of our model and the average roll of the die as we roll more and more times.
        - What is the bias of a model that alway predicts 3? 
   


```python
from src.student_caller import one_random_student
from src.student_list import student_first_names

%load_ext autoreload
%autoreload 2
```


```python
one_random_student(student_first_names)
```

   - The variance is the average difference between each individual prediction and the average prediction of our model as we roll more and more times.
        - What is the variance of that model?


```python
one_random_student(student_first_names)
```

# 2. Defining Error: prediction error and irreducible error



### Regression fit statistics are often called “error”
 - Sum of Squared Errors (SSE)
 $ {\displaystyle \operatorname {SSE} =\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.} $
 - Mean Squared Error (MSE) 
 
 $ {\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.} $
 
 - Root Mean Squared Error (RMSE)  
 $ {\displaystyle \operatorname 
  {RMSE} =\sqrt{MSE}} $

 All are calculated using residuals    

![residuals](img/residuals.png)


## This error can be broken up into parts:

![defining error](img/defining_error.png)

There will always be some random, irreducible error inherent in the data.  Real data always has noise.

The goal of modeling is to reduce the prediction error, which is the difference between our model and the realworld processes from which our data is generated.

# 3. Define prediction error as a combination of bias and variance

$\Large Total\ Error\ = Prediction\ Error+ Irreducible\ Error$

Our prediction error can be further broken down into error due to bias and error due to variance.

$\Large Total\ Error = Model\ Bias^2 + Model\ Variance + Irreducible\ Error$



**Model Bias** is the expected prediction error of the expected trained model

> In other words, if you were to train multiple models on different samples, what would be the average difference between the prediction and the real value.

**Model Variance** is the expected variation in predictions, relative to your expected trained model

> In other words, what would be the average difference between any one model's prediction and the average of all the predictions .



# Thought Experiment

1. Imagine you've collected 23 different training sets for the same problem.
2. Now imagine training one model on each of your 23 training sets.
3. Bias vs. variance refers to the accuracy vs. consistency of the models trained by your algorithm.

![target_bias_variance](img/target.png)

http://scott.fortmann-roe.com/docs/BiasVariance.html



### Let's take a look at our familiar King County housing data. 

After some EDA, we have decided to choose 11 independent features predicting 1 target variable, price.


```python
import pandas as pd
import numpy as np
df = pd.read_csv('data/king_county.csv', index_col='id')
df = df.iloc[:,:12]
df.head()
```

Let's create a set of 100 trained models by randomly selecting 1000 records, and look at the difference in predictions w.r.t. 1 point.


```python
np.random.seed(11)

# Reserve a random sample point for demonstration of bias/variance 
sample_point = df.sample(1)
true_sample_price = sample_point.price
```


```python
sample_point.drop('price', axis=1, inplace=True)

print(sample_point.head())
```


```python
print(f'Sample home price {true_sample_price.values[0]}')

```


```python
# Remove sample from data set we will train our model on
df.drop(true_sample_price.index[0], axis=0, inplace=True)
print(df.shape)
```


```python
### from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
np.random.seed(11)

# Let's generate random subsets of our data

point_preds_simp = []
simple_rmse = []

for i in range(100):
    
    # Sample 1000 random homes
    df_sample = df.sample(1000, replace=True)
    y = df_sample.price
    X = df_sample.drop('price', axis=1)
    
    # Create a trained model for each subset
    lr = LinearRegression()
    lr.fit(X, y)
    
    y_hat = lr.predict(X)
    
    # Calculate RMSE for each trained model
    simple_rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    
    # Predict a value for the sample point
    y_hat_point = lr.predict(sample_point)
    point_preds_simp.append(y_hat_point)
    

```

Now let's use sklearn's polynomial transformation to create a relatively complex version of our model.  
[Poly_transform blog](https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/)



```python
from sklearn.preprocessing import PolynomialFeatures

# This will create a feature set of each feature squared, as well as interaction features between each independent variable.
pf = PolynomialFeatures(2, include_bias=False)

df_poly = pd.DataFrame(pf.fit_transform(df.drop('price', axis=1)))
df_poly.index = df.index
df_poly['price'] = df['price']

cols = list(df_poly)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('price')))

df_poly = df_poly.loc[:,cols]

df_poly.head(10)
```


```python

# Isolate the poly-transformed version of our sample point
sample_point = pf.transform(sample_point)
sample_point
```

Then train 100 models using our complex features set on samples of size 5000.


```python
np.random.seed(11)

point_preds_comp = []
complex_rmse = []
for i in range(100):
    
    df_sample = df_poly.sample(1000, replace=True)
    y = df_sample.price
    X = df_sample.drop('price', axis=1)
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_hat = lr.predict(X)
    complex_rmse.append(np.sqrt(mean_squared_error(y, y_hat)))
    
    y_hat_point = lr.predict(sample_point)
    
    point_preds_comp.append(y_hat_point)
    
```


```python
print("#################### BIAS ###########################")
print(f'mean simple prediction      {np.mean(point_preds_simp)}')
print(f'mean complex prediction     {np.mean(point_preds_comp)}')
print(f'true value                  {true_sample_price}')
print("################## VARIANCE #########################")
print(f'simp variance {np.var(point_preds_simp)}')
print(f'comp variance {np.var(point_preds_comp)}')
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,10))

sns.violinplot(point_preds_simp, ax=ax1, orient='h', color='orange')
ax1.set_title("Simple Model")
ax1.axvline(true_sample_price.values[0])
sns.violinplot(point_preds_comp, ax=ax2, orient='h', color='yellow')
ax2.axvline(true_sample_price.values[0])
ax2.set_title("Complex Model");
```

![stretch goal](https://media.giphy.com/media/XBG7hzVQymRJk2LPpE/giphy.gif)

If you are curious after class, try fitting a 3rd order polynomial and plot the predictions w.r.t. the sample point. The mean of your predictions should align more tightly around the true value, but the variance should be much larger.


# 4.  Explore Bias Variance Tradeoff

**High bias** algorithms tend to be less complex, with simple or rigid underlying structure.

+ They train models that are consistent, but inaccurate on average.
+ These include linear or parametric algorithms such as regression and naive Bayes.
+ For linear, perhaps some assumptions about our feature set could lead to high bias. 
      - We did not include the correct predictors
      - We did not take interactions into account
      - In linear, we missed a non-linear relationship (polynomial). 
      
High bias models are **underfit**

On the other hand, **high variance** algorithms tend to be more complex, with flexible underlying structure.

+ They train models that are accurate on average, but inconsistent.
+ These include non-linear or non-parametric algorithms such as decision trees and nearest neighbors.
+ For linear, perhaps we included an unreasonably large amount of predictors. 
      - We created new features by squaring and cubing each feature
+ High variance models are modeling the noise in our data

High variance models are **overfit**



While we build our models, we have to keep this relationship in mind.  If we build complex models, we risk overfitting our models.  Their predictions will vary greatly when introduced to new data.  If our models are too simple, the predictions as a whole will be inaccurate.   

The goal is to build a model with enough complexity to be accurate, but not too much complexity to be erratic.

![optimal](img/optimal_bias_variance.png)
http://scott.fortmann-roe.com/docs/BiasVariance.html



![which_model](img/which_model_is_better_2.png)

# 5. Train Test Split


```python
from sklearn.model_selection import train_test_split

```

It is hard to know if your model is too simple or complex by just using it on training data.

We can hold out part of our training sample, and use it as a test sample and use it to monitor our prediction error.

This allows us to evaluate whether our model has the right balance of bias/variance. 

<img src='img/testtrainsplit.png' width =550 />

* **training set** —a subset to train a model.
* **test set**—a subset to test the trained model.



```python
import pandas as pd
df = pd.read_csv('data/king_county.csv', index_col='id')

y = df.price
X = df[['bedrooms', 'sqft_living']]

# For test size, we generally choose a number between .2 and .3.  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = .25)

print(X_train.shape)
print(X_test.shape)

print(X_train.shape[0] == y_train.shape[0])
print(X_test.shape[0] == y_test.shape[0])
```

**How do we know if our model is overfitting or underfitting?**


If our model is not performing well on the training  data, we are probably underfitting it.  


To know if our  model is overfitting the data, we need  to test our model on unseen data. 
We then measure our performance on the unseen data. 

If the model performs way worse on the  unseen data, it is probably  overfitting the data.

<img src='https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg' width=500/>

# Word Play

Fill in the variable to correctly finish the sentences.



```python
one_random_student(student_first_names)
```


```python

b_or_v = 'add a letter'
over_under = 'add a number'

one = "The model has a high R^2 on the training set, but low on the test " +  b_or_v + " " + over_under
two = "The model has a low RMSE on training and a low RMSE on test" + b_or_v + " " + over_under
three = "The model performs well on data it is fit on and well on data it has not seen" + b_or_v + " " + over_under
seven = "The model has high R^2 on the training set and low R^2 on the test"  + b_or_v + " " + over_under
four = "The model leaves out many of the meaningful predictors, but is consistent across samples" + b_or_v + " " + over_under
five = "The model is highly sensitive to random noise in the training set"  + b_or_v + " " + over_under
six = "The model has a low R^2 on training but high on the test set"  + b_or_v + " " + over_under


a = "The model has low bias and high variance."
b = "The model has high bias and low variance."
c = "The model has both low bias and variance"
d = "The model has high bias and high variance"

over = "In otherwords, it is overfit."
under = "In otherwords, it is underfit."
other = 'That is an abberation'
good = "In otherwords, we have a solid model"

print('###############One##################')
print(one)
print('###############Two##################')
print(two)
print('##############Three#################')
print(three)
print('##############Four##############')
print(four)
print('##############Five#################')
print(five)
print('##############Six################')
print(six)



```

### Should you ever fit on your test set?  


![no](https://media.giphy.com/media/d10dMmzqCYqQ0/giphy.gif)


**Never fit on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. 



Let's go back to our KC housing data without the polynomial transformation.


```python
df = pd.read_csv('data/king_county.csv', index_col='id')

#Date  is not in the correct format so we are dropping it for now.
df.head()
```

Now, we create a train-test split via the sklearn model selection package.


```python
from sklearn.model_selection import train_test_split
np.random.seed(42)

y = df.price
X = df.drop('price', axis=1)

# Here is the convention for a traditional train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=.25)
```


```python
# Instanstiate your linear regression object
lr = LinearRegression()
```


```python
# fit the model on the training set
lr.fit(X_train, y_train)
```


```python
# Check the R^2 of the training data
lr.score(X_train, y_train)
```


```python
lr.coef_
```

A .513 R-squared reflects a model that explains aabout half of the total variance in the data. 

### Knowledge check
How would you describe the bias of the model based on the above training R^2?


```python
# Your answer here
```

Next, we test how well the model performs on the unseen test data. Remember, we do not fit the model again. The model has calculated the optimal parameters learning from the training set.  



```python
lr.score(X_test, y_test)
```

The difference between the train and test scores are low.

What does that indicate about variance?

# Now, let's try the same thing with our complex, polynomial model.


```python
df = pd.read_csv('data/king_county.csv', index_col='id')
df.head()
```


```python
poly_2 = PolynomialFeatures(3, include_bias=False)

X_poly = pd.DataFrame(
            poly_2.fit_transform(df.drop('price', axis=1))
                      )

y = df.price
X_poly.head()
```


```python
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=20, test_size=.25)
lr_poly = LinearRegression()

# Always fit on the training set
lr_poly.fit(X_train, y_train)

lr_poly.score(X_train, y_train)
```


```python
# That indicates a lower bias
```


```python
lr_poly.score(X_test, y_test)
```


```python
# There is a large difference between train and test, showing high variance.
```

# Pair Exercise

##### [Link](https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data) about data leakage and scalars

The link above explains that if you are going to scale your data, you should only train your scaler on the training data to prevent data leakage.  

Perform the same train test split as shown aboe for the simple model, but now scale your data appropriately.  

The R2 for both train and test should be the same.



```python
from sklearn.preprocessing import StandardScaler
np.random.seed(42)

y = df.price
X = df.drop('price', axis=1)

# Train test split with random_state=43 and test_size=.25

# Instantiate an instance of Standard Scaler, and fit/transform on the training data

# Transform the test data with the fit scalar

# fit and score the model 

```

# Kfolds: Even More Rigorous Validation  

For a more rigorous cross-validation, we turn to K-folds

![kfolds](img/k_folds.png)

[image via sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)

In this process, we split the dataset into train and test as usual, then we perform a shuffling train test split on the train set.  

KFolds holds out one fraction of the dataset, trains on the larger fraction, then calculates a test score on the held out set.  It repeats this process until each group has served as the test set.

We tune our parameters on the training set using kfolds, then validate on the test data.  This allows us to build our model and check to see if it is overfit without touching the test data set.  This protects our model from bias.

# Fill in the Blank


```python
from src.student_caller import one_random_student, three_random_students
from src.student_list import student_first_names
three_random_students(student_first_names)
```


```python
X = df.drop('price', axis=1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=.25)

```


```python
from sklearn.model_selection import KFold

# Instantiate the KFold object
kf = KFold(n_splits=5)

train_r2 = []
test_r2 = []

# kf.split() splits the data via index
for train_ind, test_ind in kf.split(X_train,y_train):
    
    X_tt, y_ttt = fill_in, fill_in
    X_val, y_val = fill_in, fill_in
    
    # fill in fit
    
    
    train_r2.append(lr.score(X_tt, y_tt))
    test_r2.append(lr.score(X_val, y_val))
```


```python
# Mean train r_2
np.mean(train_r2)
```


```python
# Mean test r_2
np.mean(val_r2)
```


```python
# Test out our polynomial model
poly_2 = PolynomialFeatures(2)

df_poly = pd.DataFrame(
            poly_2.fit_transform(df.drop('price', axis=1))
                      )

X = df_poly
y = df.price

```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=.25)

kf = KFold(n_splits=5)

train_r2 = []
val_r2 = []
for train_ind, test_ind in kf.split(X_train,y_train):
    
    X_tt, y_tt = X_train.iloc[train_ind], y_train.iloc[train_ind]
    X_val, y_val = X_train.iloc[test_ind], y_train.iloc[test_ind]
    
    lr.fit(X_tt, y_tt)
    train_r2.append(lr.score(X_tt, y_tt))
    val_r2.append(lr.score(X_val, y_val))
```


```python
# Mean train r_2
np.mean(train_r2)
```


```python
# Mean test r_2
np.mean(val_r2)
```

By using this split, we can use the training set as a test ground to build a model with both low bias and low variance.
We can test out new independent variables, try transformations, implement regularization, up/down sampling, without introducing bias into our model.

Once we have an acceptable model, we train our model on the entire training set, and score on the test to validate.




```python
lr_final = LinearRegression()
lr_final.fit(X_train, y_train)

lr_final.score(X_test, y_test)
```


```python

```
