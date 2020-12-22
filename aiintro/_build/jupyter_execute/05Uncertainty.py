# Modelling of Uncertainty
* Author: Johannes Maucher
* Last Update: 26.10.2020

## Motivation

<img src="https://maucher.home.hdm-stuttgart.de/Pics/cancerSlide.png" style="width:700px" align="center">

**Motivation:**

* This example demonstrates <font color="red"> how intuition can mislead us</font>.
* The book [D. Kahnemann, Thinking Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) is full of such examples, where we estimate probabilities and risks just by intuition.
* Estimation of risks (e.g. in the planning of complex projects) is hard, because of the interaction of many non-deterministic factors
* Bayesian Networks help to model such complex problems.
* They <font color="red">allow global probability calculation, based on local (pairwise) probability estimates.</font> 

## Estimation of Probabilities from a sample

import numpy as np
import pandas as pd
from IPython.display import display
np.random.seed(1235)

P=np.random.randint(0,2,(10,3))
idx=pd.index=["user"+str(i+1) for i in range(10)]
cols=["ipad","iphone","macbook"]
userdata=pd.DataFrame(data=P,index=idx,columns=cols)

display(userdata)

**A-priori probabilities:**

$P(ipad)=\frac{5}{10}=0.5$

$P(iphone)=\frac{3}{10}=0.3$

$P(macbook)=\frac{4}{10}=0.4$

**Compound probabilities:**

$P(ipad,iphone)=\frac{2}{10}=0.2$

$P(ipad,macbook)=\frac{3}{10}=0.3$

$P(iphone,macbook)=\frac{2}{10}=0.2$

**Conditional probabilities:**

$P(ipad | iphone)=\frac{2}{3}=0.66$

$P(ipad | iphone,macbook)=\frac{2}{2}=1.0$

$P(iphone | ipad)=\frac{2}{5}=0.4$

## Bayesian Network
* Bayesian Networks
   * constitute a <font color = "red">knowledge base</font>, where knowledge is represented in the language of probabilities
* allow inference in both directions: 
    - causal
    - diagnostic

**Applications:**

* Insurance companies: Risk assessment and -estimation
* Medical Diagnosis
* Technical Diagnosis
* Situation Analysis (Project with Daimler). E.g. given the input of many sensors, estimate the probability that a neighbouring car changes lines.


**Bayesian Networks:**

Defined by:
* <font color="red">Nodes</font>: Variables
* <font color="red">Edges</font>: Direct dependencies between variables
* <font color="red">Conditional Probability Tables (CPT)</font> at each node: Contain the conditional probabilities of the node-variable, given the parent-variables.

### Configuration of Network
<img src="https://maucher.home.hdm-stuttgart.de/Pics/bayesNetAsia.PNG" style="width:600px" align="center">

Bayesian Network is configured by
* drawing a node for each relevant random variable
* connecting directly dependent variables by edges
    - pairs of independent variables need not be connected
    - pairs of conditional independent variables need not be connected
* Configuration of the CPTs for each node. The conditional probabilities to be inserted in the CPTs are estimated, e.g. from samples, expert knowledge, scientific studies, ...

**Example:**

In the introduction the random variables are 
* Cancer (C)
* Mammography (M)

The CPT at random variable $C$ consists of the single entry 
* $P(C)=0.01$

The CPT at random variable $M$ consists of
* $P(M|C)=0.8$
* $P(M|\neg C)=0.096$

The task is to determine $P(C|M)$

<img src="https://maucher.home.hdm-stuttgart.de/Pics/bayesNetCancer.PNG" style="width:600px" align="center">

Given the observation *Mammography positiv*, the probability for *Cancer* is 7.76%.

<img src="https://maucher.home.hdm-stuttgart.de/Pics/bayesNetAsiaInfer.PNG" style="width:600px" align="center">

Given the observation *Smoker positiv* and *X-Ray positive* the probability for e.g. *Dyspnoea* is 73.19% and the probability for *lung-cancer* is 64.60%.

## Modelling of Risks in Project Management
<img src="https://maucher.home.hdm-stuttgart.de/Pics/bayesNetProjectMgmt.PNG">

### Causal Inference
<div class="text-medium container">
    <p>Project Manager realizes lack of communication (FehlendeKommunikation)</p>
    <p>How does this observation impact other factors?</p>
</div>
<img src="https://maucher.home.hdm-stuttgart.de/Pics/bayesNetProjectMgmtKommunikation.PNG">

### Diagnostic Inference
<div class="text-medium container">
    <p>The scheduled time could not be met. </p>
    <p>What is the most likely reason, if it is known that the requirements were correctly defined from the beginning?</p>
</div>
<img src="https://maucher.home.hdm-stuttgart.de/Pics/bayesNetProjectMgmtDiagnose.PNG">