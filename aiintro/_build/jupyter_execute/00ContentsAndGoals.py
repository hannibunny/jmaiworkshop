# AI Seminar Mahle
* Author: Prof. Dr. Johannes Maucher
* Date: 09.10.2020
* Location: 

## Overall Goals
This workshop consists of 3 blocks, which are described below. The overall goals are: 

* Artificial Intelligence (AI) is not about creating super-human-science-fiction-creatures. AI, instead provides an extremely flexible and comprehensive toolbox, which can be applied for an almost infinite field of applications. The main goal of this workshop is to provide you a glimpse into the immense AI potential. In the best case you will find out, that some of your daily work or projects can be better solved by **applying AI-tools.**

* **Just do it!** - You should not be afraid of AI and it's supposed complexity. Instead immediately start your first project after this seminar by adapting the provided Jupyter-notebooks to your task and data. You will find out, that there are solutions (Jupyter Notebooks) for a vast variety of different problems on GitHub, Bitbucket, etc. If you manage to **abstract your problem**, you will find a solution, which can be adapted to solve your problem.
 


## Block 1: AI categories and some sample applications
The field of Artificial lIntelligence is usually partitioned into the following 4 categories:

* Search and Plan
* Knowledge Representation and Inference
* Modelling of Uncertainty
* Machine Learning

In this first block the general concepts of these categories are described. For each of the categories some representative applications are discussed. 

The first 3 categories will be sketched only on a shallow abstract level. The fourth category **Machine Learning** is currently by far the most important one. Actually, if people speak about AI they almost always mean Machine Leraning and in particular **Deep Learning**. Hence, in the following two blocks we focus only on Machine Learning and Deep Learning.

## Block 2: Implementation of Data Mining Process in Python

Machine Learning algorithms are at the heart of any Data Mining process. However, the entire Data Mining process consists of other modules:
1. Data Access
2. Data Understanding and Selection
3. Data Cleaning
4. Data Transformations
5. Modelling (<- that's the Machine Learning part)
6. Evaluation
7. Visualisation and Interpretation

In this block the entire Data Mining process chain is implemented in Python and the following Python packages:
* [numpy](http://www.numpy.org) and [scipy](https://www.scipy.org) for efficient datastructures and scientific calculations
* [pandas](https://pandas.pydata.org) for typical data science tasks, such as data access, descriptive statistics, joining of datasets, correlation analysis, etc.
* [matplotlib](https://matplotlib.org) and [Bokeh](https://bokeh.pydata.org/en/latest/)
* [scikit-learn](https://scikit-learn.org/stable/) for conventional Machine Learning. I.e. all but Deep Neural Networks
* [tensorflow](https://www.tensorflow.org) and [keras](https://keras.io) for Deep Neural Networks

## Block 3: Neural Networks and their Implementations and Applications in Python

Almost all of the recently popular AI applications, such as AlphaZero, Object Recognition, Speech Recognition, Language Understanding, Automatic Translation, Style-Transfer, Image-Captioning, Automatic generation of text, speech, audio, images, video, ... are Deep Neural Networks. 

In this block the basics of conventional neural networks are presented. Then the most important concepts, layer-types and architectures of deep neural networks are introduced. Finally it is shown how deep neural network applications can be implemented in a fast and efficient way applying [tensorflow](https://www.tensorflow.org) and [keras](https://keras.io). Jupyter Notebooks, e.g. for Object Recognition, Document Classification and Time Series Prediction are provided.   