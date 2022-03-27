---
title: 'machine learning introduction'
metaTitle: 'High level overview of the ML landscape'
metaDesc: 'What it is, how it works, and why we do it'
socialImage: images/introml.jpeg
date: '2022-03-26'
tags:
  - ml
---
March 26, 2022

## What is Machine Learning?

Machine learning is the art and science of programming computers so that they can learn from data.

> Giving computers the ability to learn without explicitly being programmed.
> 

An example is a spam filter program, which learns to flag emails given many examples of spam and regular emails.

The **training set** is examples which the system uses to learn. Each training example is called an **instance** or **sample**.

## Why use Machine Learning?

Consider how traditional algorithms / programs are written.

1. Study the problem
    1. Identify patterns in the data (spam emails usually contain “free”, “amazing”, etc.)
2. Write rules for the program
    1. Write detection cases to cover each of the patterns noticed
3. Test the program
    1. Analyze the accuracy of the test cases passed and failed
    2. Repeat steps 1 and 2 until the algorithm performs well enough

For less trivial problems, such as spam detection, this can become a very laborious process of writing a huge set of conditions. This is not efficient or fun to maintain.

![https://images.unsplash.com/photo-1485546246426-74dc88dec4d9?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb](https://images.unsplash.com/photo-1485546246426-74dc88dec4d9?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb)

When Machine Learning is used, the program learns on its own which features in the data are good predictors of the expected output. 

- For example, it would learn that the words “credit card” are a good predictor of a spam email.

By creating machine learning models which learn rules / patterns in data on their own, our programs can be much *shorter*, *easier to maintain*, and *accurate*.

- If spammers started changing the words “credit card” to “cr3d!t c@rd” to avoid being flagged, the ML model would notice the frequency of the keywords in spam emails and update its weights.

> Machine learning shines in problems which are too complex for traditional approaches, or have no known algorithm
> 

Applying ML approaches to large amounts of data to discover new patterns is called **data mining**.

![https://images.unsplash.com/photo-1581878611345-3fe425a0f833?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb](https://images.unsplash.com/photo-1581878611345-3fe425a0f833?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb)

## Types of Machine Learning Systems

Machine learning systems can be classified into broad categories:

- supervised, unsupervised, semisupervised, and reinforcement learning
- online learning, batch learning
- instance-based learning, model-based learning

### Supervised Learning

Supervised learning involves training an algorithm on **labeled data**. This is data which includes solutions, such as a set of pictures with captions of them.

**Classification** is the task of predicting a class of a certain data sample.

- Predicting if an image is a dog or cat
- Predicting if a handwritten letter is “B” or “P”.

**Regression** is the task of predicting a numeric value.

- Given the year and mileage of a car, what is its dollar value?

Some important supervised learning algorithms include:

- [ ]  k-Nearest neighbors
- [ ]  Linear regression
- [ ]  Logistic regression
- [ ]  Support vector machines (SVMs)
- [ ]  Decision trees and Random forests
- [ ]  Neural networks

### Unsupervised Learning

Unsupervised learning involves drawing patterns from data which is **unlabeled**.

Some important unsupervised learning approaches:

**Clustering**

- [ ]  k-Means
- [ ]  Hierarchical Cluster Analysis
- [ ]  Expectation Maximization

**Visualization and dimensionality reduction**

- [ ]  Principal Component Analysis (PCA)
- [ ]  Kernel PCA
- [ ]  Locally-Linear Embedding (LLE)
- [ ]  t-distributed Stochastic Neighbor Embedding (t-SNE)

**Association rule learning**

- [ ]  Apriori
- [ ]  Eclat

**Clustering** algorithms try to detect groups within your data set.

**Visualization** algorithms output a 2D or 3D representation of unlabeled data which can be easily plotted. They help to understand how data is organized and to identify unsuspected patterns.

**Dimensionality reduction** deals with simplifying data without losing too much information.

- Can be done by merging several correlated features into one
- A part of ***feature extraction***
- Reducing the dimension of training data before feeding into another ML algorithm is key to faster training and using less memory  / compute.

**Anomaly detection** focuses on finding anomalies among data sets.

- useful to prevent fraud in credit card transactions

**Association rule learning** involves analyzing large amounts of data to find interesting relationships between attributes / features

- Running association rule on store sales logs may reveal association of buyers purchasing bbq sauce, potato chips, and steak together, suggesting these should be located together.

### Semisupervised Learning

Semisupervised learning deals with partially labeled data, lots of unlabeled data, and some labeled data.

Most of these algorithms are combinations of unsupervised and supervised algorithms.

- Deep Belief Networks (DBNs) are based on unsupervised components (Restricted Boltzmann Machines) stacked on top of each other. These RBMs are trained sequentially, unsupervised, then the whole system is fine tuned using supervised techniques.

### Reinforcement Learning

A way to think of RL is learning by doing, similar to how babies learn to walk.

The learning system is called an **agent**, which can observe its **environment**, select and perform **actions**, and potentially get a **reward** in return. 

The agent’s goal is to learn the best strategy, or **policy**, to maximize reward over time.

The **policy** defines what action an agent should take given its situation.

![https://images.unsplash.com/photo-1589254065878-42c9da997008?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb](https://images.unsplash.com/photo-1589254065878-42c9da997008?ixlib=rb-1.2.1&q=85&fm=jpg&crop=entropy&cs=srgb)

Many robots use RL to learn how to walk / maneuver in their environments.

RL is also used extensively when training agents to play games, such as Go or chess.

## Batch Learning and Online Learning

### Batch Learning

In batch learning, systems cannot learn incrementally. Systems are trained using all available data which takes lots of time and compute.

**Offline Learning**

After systems are finished with training offline, they are launched into production and don’t learn anymore. They simply apply their previous learnings to the new data. 

To train a batch learning system on new data, a new version of the system must be trained from scratch on all of the data.

In short, it is not efficient to have to retrain the model on the entire data set when new data needs to be considered. This is why large scale systems with lots of data prefer to learn incrementally, with online learning.

### Online Learning

In online learning, systems can be trained incrementally by feeding them data instances sequentially. Learning steps are fast and cheap, and the system learns new data as soon as it arrives.

This is great for systems which receive data as a continuous flow (stock data, weather data, user data, etc). 

Online learning is also ideal for situations with limited computing resources, since data instances are not needed once they are learned by the system.

The **Learning rate** determines how fast the system should adapt to changing data.

A high learning rate will cause the system to adapt to new data and quickly forgett old data.

A low learning rate will learn more slowly, and will be less sensitive to noise in the new data.

## Instance-Based Learning and Model-Based Learning

Machine learning systems can be categorized by how they *generalize*.

Most ML tasks are about **making predictions** based on previously seen training data. 

As such, a system must learn to **generalize** its learnings to examples it hasn’t seen before (test data). 

The goal is to have a model perform well on new instances (test set) in addition to performing well on the training set.

### Instance-Based Learning

Instance based learning focuses on learning the training examples by heart, then generalizing to new cases based on their similarity to known cases.

- An example would be classifying a new email as spam or not based on the number of words it has in common with previous examples of spam emails.
- Predictions on new examples are made by ***measuring similarity*** between new examples and known examples.

### Model-Based Learning

Building a model based on training examples, then using that model to make new predictions.

This is arguably the best, and most popular approach of the two.

The steps to model based learning are:

1. Study the data
2. Select a model
3. Train the model on training data
    - The learning algorithm will search for model parameters which minimize the cost function
4. Apply the model to make predictions on new cases (**inference**)
    - Observe how well the model *generalizes*