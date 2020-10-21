# Rational Agents
* Author: Johannes Maucher
* Last Update: 15.10.2020

**Goals of this section:**
* Raise awareness of the essential factors of (artificial) intelligence
* Understand AI definition of [S. Russel and P. Norvig](http://aima.cs.berkeley.edu)
* This definition yields a <font color="red">process model</font> for a structured development of any AI-project $\Rightarrow$ <font color="red">Learn to proceed according to this model.</font>

## What is Intelligence?

## What is Artificial Intelligence?

AI develops ...

<img src="https://maucher.home.hdm-stuttgart.de/Pics/aiDefinitions.png" style="width:400px">

Engineer's definition ([S. Russel and P. Norvig](http://aima.cs.berkeley.edu)): <font color = red>*AI develops rational agents*</font>

### Rational Agents

![agentGeneral.png](https://maucher.home.hdm-stuttgart.de/Pics/agentGeneral.png)

A rational agent is anything that is
1. **perceiving** its environment through sensors
2. **thinking** and deciding on the next actions
2. **acting** through actuators

Rational means, that the agent acts in a way that is expected to maximize its performance measure, given it's
* built-in knowledge
* perceived experience
* acting capabilities 

## Specification of Environment

1. AI-specific specification of problem domain
2. Determines which <font color="red">agent type</font> is required
3. Agent type determines 
    - required modules in the agent
    - algorithm categories required in the agent

### Environment consists of
* **P**erformance Measure
* **E**nvironment
* **A**ctors
* **S**ensors

### Environment Attributes
* fully or partially **observable**
* **Deterministic** or **Stochastic**
* **Episodic** or **Sequential**
* **Static**, **Semi-Static** or **Dynamic**
* **Single-** or **Multi-Agent**

## Agent Types

### Simple-Reflex Agent 

![agentGeneral.png](https://maucher.home.hdm-stuttgart.de/Pics/agentRule.png)

The **Simple-Reflex-Agent** seems quite unintelligent. It applies it's set of rules on the current perceived state in order to select an action. Note that this type of agent even does not have a memory. Example:
* theromstat
* simple web-site, which reacts on klicks and provides new views, pages, etc.

### Model-based Agent
![agentModel.png](https://maucher.home.hdm-stuttgart.de/Pics/agentModel.png)

If the environment is not fully-observable, the non-observable part must be modelled. The model may describe how
- the environment behaves independent of the agent (external state)
- how the agent behaves in the environment (internal state)

Examples:
- In recommender systems users are modelled by their previous purchases
- In autonomous driving movement of other vehicles, pedestrians, ... is modelled
- Search engines model the user by their click-history, location, ... 
- Cleaning robots may learn a dirt-distribution model of rooms

### Goal-based Agent
![agentGoal.png](https://maucher.home.hdm-stuttgart.de/Pics/agentGoal.png)

In Goal-based agents one or a set of goals are given. The task is typically to find/plan a sequence of actions, which efficiently lead from the current state to a goal-state. Note that the planning of actions is done offline. Only after a path to the goal has been found, the corresponding actions are executed. 

Examples:
* Pathfinding, Navigation
* Planning in board-games such as checkers, chess, ...

### Utility-based Agent
![agentUtility.png](https://maucher.home.hdm-stuttgart.de/Pics/agentUtility.png)

Utility-based agents are similar to goal-based agents. They are applicable if 
* concrete goals can hardly be defined, 
* if many goals exist,
* if a complete planning from the current state to a goal state may be too complex,

Utility based agents can bea applied for all **kinds of optimization problems**. Prerequisite is the definition of a utility-function.

Examples:
* In board-games like chess planning to the end is way to complex. Instead planning is done only for a predefined number of next moves (planning horizon). Then a utility-function is required to evaluate all states, which are reachable within the planning horizon.
* Logistic
* Scheduling
* Network coverage 


### Knowledge-based Agent
![agentKnowledge.png](https://maucher.home.hdm-stuttgart.de/Pics/agentKnowledge.png)

Knowledge-based agents infere their actions, based on a comprehensive knowledge-base. There may exist some in-built-knowledge, but knowledge increases while the agent acts in it's environments and perceives new experience. Key-elements of this type of agents are:
* Formal knowledge representation, e.g. by knowledge graphs, first-order-logic, probabilities, 
* Ways to infere new knowledge and actions (logic solvers, Bayes-Net, etc.)

Example:
* Expert-Systems for e.g. medical or technical diagnosis
* IBM-Watson

### Learning Agent
![agentLearning.png](https://maucher.home.hdm-stuttgart.de/Pics/agentLearn.png)

* **Performing element** is any of the agent types, described above. E.g. if it is 
    - a simple-reflex agent, the rules can be learned
    - a model-based agent, the model can be learned
    - a goal-based agent, the goals can be adapted
    - ...
* The **Compare**-block evaluates the current perceived state with respect to the performance measure and provides negative or positive feedback to the learning element.
* Depending on the received feedback, the **Learning-Element** adapts elements of the Performing Element.
* The **Explore** element suggests actions, which do not fully **exploit** the current available knowledge, but enables the agent to gather new **experience**

