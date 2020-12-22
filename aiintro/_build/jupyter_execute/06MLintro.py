# Machine Learning: Introduction

* Author: Johannes Maucher
* Last Update: 26.10.2020

## Definition
**Definition by Tom Mitchell:**
*A computer program is said to learn from <font color="red">experience E</font> with respect to some class of <font color="red">tasks T</font> and <font color="red">performance measure P</font> if its performance at tasks in T, as measured by P, improves with experience E.*

Similar as the definition of *rational Agents* for AI in general, this definition of Machine Learning defines a practical guideline to start any new project in this field: At the very beginning of each new project in this field specify the task T, the performance measure P and the experience E. For some applications some of these elements may be trivial, but sometimes it can be very challenging. E.g. what is the performance measure for a recommender system? 

### Example: Recommender System Online Shop

1. Task T:
    * ?
2. Experience E:
    * ?
3. Performance Measure P:
    * ?

## Categories of Machine Learning
<img src="http://maucher.home.hdm-stuttgart.de/Pics/mlCategories.png" style="width:800px" align="center">

The field of Machine Learning is usually categorized with respect to two dimensions: The first dimension is the question *What shall be learned?* and the second asks for *How shall be learned?*. The resulting 2-dimensional matrix is depicted above.

On an abstract level there exist 4 answers on the first question. One can either learn 

* **a classifier**, e.g. object recognition, spam-filter, Intrusion detection, ...
* **a regression-model**, e.g. time-series prediction, like weather- or stock-price forecasts, range-prediction for electric vehicles, estimation of product-quantities, ...
* **associations between instances**, e.g. document clustering, customer-grouping, quantisation problems, automatic playlist-generation, ....
* **associations between features**, e.g. market basket analysis (customer who buy cheese, also buy wine, ...)
* **strategie**, e.g. for automatic driving or games 

![ClassificationRegression](http://maucher.home.hdm-stuttgart.de/Pics/classReg.PNG)

On the 2nd dimension, which asks for *How to learn?*, the answers are:

* **supervised learning:** This category requires a *teacher* who provides labels (target-values) for each training-element. For example in face-recognition the teacher most label the inputs (pictures of faces) with the name of the corresponding persons. In general labeling is expensive and labeled data is scarce. 
* **unsupervised learning:** In this case training data consists only of inputs - no teacher is required for labeling. For example pictures can be clustered, such that similar pictures are assigned to the same group.
* **Reinforcement learning:** In this type no teacher who lables each input-instance is available. However, there is a critics-element, which provides feedback from time-to-time. For example an intelligent agent in a computer game maps each input state to a corresponding action. Only after a possibly long sequence of actions the agent gets feedback in form of an increasing/decreasing High-Score.  

### Supervised Learning
<img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearning.png" style="width:800px" align="center">

**Apply Learned Modell:**
<img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearningApply.png" style="width:800px" align="center">

### Unsupervised Learning
<img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearningUnsupervised.png" style="width:800px" align="center">

**Apply learned Model:**
<img src="http://maucher.home.hdm-stuttgart.de/Pics/introExampleLearningUnsupervisedApply.png" style="width:800px" align="center">

### Reinforcement Learning
<img src="https://maucher.home.hdm-stuttgart.de/Pics/bogenschiessen.jpg" style="width:500px" align="center">

## Categorization on Application-Level
<img src="http://maucher.home.hdm-stuttgart.de/Pics/MLapplicationCategories.png" style="width:800px" align="center">

## General Process- and Datascheme

### Training- and Operation Mode
<img src="http://maucher.home.hdm-stuttgart.de/Pics/SupervisedLearningSchemaEnglish.png" style="width:700px" align="center">

### Validation
<img src="http://maucher.home.hdm-stuttgart.de/Pics/SupervisedLearningSchemaValidation.png" style="width:700px" align="center">

In Machine Learning one distinguishes  
* training-mode, 
* validation-mode 
* operational mode.

In the training phase training-data is applied to learn a general model. The model either describes the structure of the training data (in the case of unsupervised learning) or a function, which maps input-data to outputs. Once this model is learned it can be applied in the operational phase to map new input-instances to output values (classes-index, cluster-index or numeric function-value). Before applying a learned model in operation it must be validated. In the case of supervised learning validation compares for all test-data the output of the model with the target output. This means that testing also requires labeled data. Test-data and training-data must be disjoint.

As shown in the picture above, usually the available data can not be passed directly to the machine-learning algorithm. Instead it must be processed in order to transform it to a corresponding format and to extract meaningful features. The usual formal, accepted by all machine-learning algorithms is a 2-dimensional array, whose rows are the instances (e.g. documents, images, customers, ...) and whose columns are the features, which describe the instances (e.g. words, pixels, bought products, ...): 



### Data Structure

Each ML algorithm requires numeric vectors of unique length at it's input. Each vector represents an **instance**, which itself is described by a set of $K$ **features**. Usually we have many such vectors for training and testing and by stacking these vectors together we end up at the following two-dimensional data array. This is the standard data-structure for ML.

<img src="http://maucher.home.hdm-stuttgart.de/Pics/dataMatrix.png" style="width:500px" align="center">

It can be challenging to transform the given data into this format. Some examples are given below. 

<img src="http://maucher.home.hdm-stuttgart.de/Pics/mlDataExamples.png" style="width:700px" align="center">

* In object recognition the instances are images and the features are all pixel-values of the image. An image of size $r \times c$ with $z$ channels is then described by $K=r \cdot c \cdot z$ features.
* In document classification a common form of representation (the *Bag-of-Word (BoW)* model) is to descrive each document (row in the matrix) by the words it contains, i.e. the columns of the 2-dimensional data-structure are the words of the entire vocabulary and the entries in this 2-dimensional indicate how often a word appears in the corresponding document
* for a recommender-system of the online-shop the instances are the customers and each customer is described by the products he or she already purchased. 
* ...

## Machine Learning in the context of Data Mining
![Crisp](http://maucher.home.hdm-stuttgart.de/Pics/crispIndall.png)

Data Mining is maybe the most frequent application of Machine Learning. Therefore the entire data mining process, which applies Machine Learning for *Modeling* is sketched here:

The **Cross-industry standard process for data mining (CRISP)** proposes a common approach for realizing data mining projects. This approach is sketched in the picture above.


In the first phase of CRISP the overall business-case, which shall be supported by the data mining process must be clearly defined and understood. Then the goal of the data mining project itself must be defined. This includes the specification of metrics for measuring the performance of the data mining project. 

In the second phase data must be gathered, accessed, understood and described. Quantitiy and qualitity of the data must be assessed on a high-level. 

In the third phase data must be investigated and understood more thoroughly. Common means for understanding data are e.g. visualization and the calculation of simple statistics. Outliers must be detected and processed, sampling rates must be determined, features must be selected and eventually be transformed to other formats.  

In the modeling phase various algorithms and their hyperparameters are selected and applied. Their performance on the given data is determined in the evaluation phase. 

The output of the evaluation is usually fed back to the first phases (business- and data-understanding). Applying this feedback the techniques in the overall process are adapted and optimized. Usually only after several iterations of this process the evaluation yields good results and the project can be deployed.

## Machine Learning Applications

**Almost all of the applications in the [Introduction](01Introduction.ipynb) are Machine Learning applications**