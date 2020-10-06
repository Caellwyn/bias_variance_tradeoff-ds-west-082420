
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

Now lets imagine we create a model that always predicts a roll of 3.

   
  - The bias is the difference between the average prediction of our model and the average roll of the die as we roll more and more times.
        - What is the bias of a model that alway predicts 3? 
   

   - The variance is the average difference between each individual prediction and the average prediction of our model as we roll more and more times.
        - What is the variance of that model?

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

Let's create a set of 100 trained models by randomly selecting 1000 records, and look at the difference in predictions w.r.t. 1 point.

Now let's use sklearn's polynomial transformation to create a relatively complex version of our model.  
[Poly_transform blog](https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/)


Then train 100 models using our complex features set on samples of size 5000.

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

It is hard to know if your model is too simple or complex by just using it on training data.

We can hold out part of our training sample, and use it as a test sample and use it to monitor our prediction error.

This allows us to evaluate whether our model has the right balance of bias/variance. 

<img src='img/testtrainsplit.png' width =550 />

* **training set** —a subset to train a model.
* **test set**—a subset to test the trained model.


**How do we know if our model is overfitting or underfitting?**


If our model is not performing well on the training  data, we are probably underfitting it.  


To know if our  model is overfitting the data, we need  to test our model on unseen data. 
We then measure our performance on the unseen data. 

If the model performs way worse on the  unseen data, it is probably  overfitting the data.

<img src='https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg' width=500/>

# Word Play

Fill in the variable to correctly finish the sentences.



```python

b_or_v = 'add a letter'
over_under = 'add a number'

a = " The model has low bias and high variance."
b = " The model has high bias and low variance."
c = " The model has both low bias and variance"
d = " The model has high bias and high variance"

over = "In otherwords, it is overfit."
under = "In otherwords, it is underfit."
other = 'That is an abberation'
good = "In otherwords, we have a solid model"


one = "The model has a low RMSE on training and a low RMSE on test." + b + " " + under
two = "The model has a high R^2 on both the training set, but low on the test." +  a + " " + over
three = "The model performs well on data it is fit on and well on data it has not seen." + c + " " + good
four = "The model has a low R^2 on training but high on the test set."  + d + " " + other
seven = "The model has high R^2 on the training set and low R^2 on the test."  + b + " " + under
four = "The model leaves out many of the meaningful predictors, but is consistent across samples." + b + " " + under
six = "The model is highly sensitive to random noise in the training set."  + a + " " + over
print(one)
print(two)
print(three)
print(four)
print(five)
print(six)


```

### Should you ever fit on your test set?  


![no](https://media.giphy.com/media/d10dMmzqCYqQ0/giphy.gif)


**Never fit on test data.** If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. 



Let's go back to our KC housing data without the polynomial transformation.

Now, we create a train-test split via the sklearn model selection package.

A .513 R-squared reflects a model that explains aabout half of the total variance in the data. 

### Knowledge check
How would you describe the bias of the model based on the above training R^2?


```python
"A model with a .608 R^2 has a relatively high bias."
```

Next, we test how well the model performs on the unseen test data. Remember, we do not fit the model again. The model has calculated the optimal parameters learning from the training set.  


The difference between the train and test scores are low.

What does that indicate about variance?


```python
'The model has low variance'
```

# Now, let's try the same thing with our complex, polynomial model.

# Pair Exercise

##### [Link](https://datascience.stackexchange.com/questions/38395/standardscaler-before-and-after-splitting-data) about data leakage and scalars

The link above explains that if you are going to scale your data, you should only train your scaler on the training data to prevent data leakage.  

Perform the same train test split as shown aboe for the simple model, but now scale your data appropriately.  

The R2 for both train and test should be the same.



```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43, test_size=.25)

ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))
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

from sklearn.model_selection import KFold

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

By using this split, we can use the training set as a test ground to build a model with both low bias and low variance.
We can test out new independent variables, try transformations, implement regularization, up/down sampling, without introducing bias into our model.

Once we have an acceptable model, we train our model on the entire training set, and score on the test to validate.


