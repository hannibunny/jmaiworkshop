#!/usr/bin/env python
# coding: utf-8

# # Introduction: What's Artificial Intelligence (AI)
# * Author: Johannes Maucher
# * Last Update: 25.11.2021

# ## Impact of AI
# 
# - *Artificial intelligence has reached a new stage of maturity in recent years and, as a basic innovation, is developing into a **driver of digitalization and autonomous systems in all areas of life**. State, society, economy, administration and science are called upon to face the chances and risks of AI.* Source: [Strategiepapier zur Künstlichen Intelligenz der Bundesregierung](https://www.ki-strategie-deutschland.de/files/downloads/201201_Fortschreibung_KI-Strategie.pdf)
# - It is supposed that AI will change life, work, economy, communication in a similar extend as the internet did before
# - AI is a cross-sectional science which
#     - enables new applications
#     - improves and optimizes existing applications
#     - improves and optimizes business processes
#     
#   in a wide range of domains. 
# 
# - [Paper on new business models enabled by AI](https://www.plattform-lernende-systeme.de/files/Downloads/Publikationen/AG4_Bericht_231019.pdf)

# ### AI is a toolbox for a wide range of problems and applications
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/LoesungsEbenen.PNG" style="width:500px" align="center">
# 
# If you manage to translate your problem into the **language of AI**, you may come up with faster and better solutions. Hence everybody should strive to learn this language. 

# ## What is Artificial Intelligence?
# ### What is intelligence?

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/intelligence.png" style="width:500px" align="center">

# #### Example: Plan Route

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/planRoute.png" style="width:700px" align="center">

# ### From Human Intelligence to Automation ...
# 
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/automatisierungOhneKI.PNG" width = "600" align="center">

# ### ... and from Automation to Machine Learning
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/automatisierungMachineLearning.PNG" width = "600" align="center">

# ### What is Artificial Intelligence?
# * **AI tries to integrate cognitive abilities**, such as inference, planning, problem solving, learning etc. in computer systems.
# * The goal of AI research is to enable modern AI systems (learning systems) such as machines, robots and software systems to **autonomously process and solve abstract tasks and problems even under changed conditions** - without a human being programming an explicit solution. 
# * All AI systems that are technically feasible today enable problem solving in limited contexts (e.g. speech or image recognition) and thus belong to the so-called **weak AI**

# #### Turing Test
# One of the most famous definitions of AI, the **Turing Test** addresses primarily a Knowledge-based agent: *The machine is considered to be intelligent if it manages to feign intelligence to humans*. 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/turingtest.jpg" style="width:400px" align="center">

# #### Categorization of AI-Definitions
# 
# AI develops ...
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/aiDefinition.png" style="width:700px">
# 
# Engineer's definition ([S. Russel and P. Norvig](http://aima.cs.berkeley.edu)): <font color = red>*AI develops rational agents*</font>

# #### Rational Agents
# 
# ![agentGeneral.png](https://maucher.home.hdm-stuttgart.de/Pics/agentGeneral.png)
# 
# See also [notebook on rational agents](01aWhatsAI.ipynb).

# A rational agent is anything that is
# 1. **perceiving** its environment through sensors
# 2. **thinking** and deciding on the next actions
# 2. **acting** through actuators
# 
# Rational means, that the agent acts in a way that is expected to maximize its performance measure, given it's
# * built-in knowledge
# * perceived experience
# * acting capabilities 

# ### AI is not new
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/aiHistory.png" style="width:800px">

# ## Categories of AI
# 
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/KiKategorien.png" style="width:400px">

# ## Some popular AI Applications 

# ### Digital Assistant
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/alexa.jpg" style="width:700px" align="middle">
# <p style="font-size: 8pt">Copyright Amazon</p>

# ### Recommender Systems
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/amazon.png" style="width:400px" align="left">
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/empfehlungssysteme_spotify.png" style="width:400px" align="right">

# ### Face-, Scene- and Objectrecognition
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/gesichtserkennung_smartphone.PNG" style="width:400px" align="middle">
# <p style="font-size: 8pt">Quelle: https://www.heise.de/mac-and-i/tipps/Gesichtserkennung-auf-iPhone-manuell-anpassen-4219849.html</p>

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/yoloLocalisationRun.png" style="width:800px" align="center">

# ### Chess, Go, Poker, ...
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/poker.jpg" style="width:500px" align="center">

# ### Early Detection of Diabetic Retinopathie
# 
# <div>
#     <img src="https://maucher.home.hdm-stuttgart.de/Pics/retinopathie-app.png" width="200" align="left">
#     <img src="https://maucher.home.hdm-stuttgart.de/Pics/retinopathie-beispiel.png" width="500" align="right">
# </div>
# 

# 
# ### Cancer Detection
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/prostate_cancer.png" width="700" align="center">
# 

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/jtHealth.png" style="width:800px">

# ### Crime
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/lkagan.PNG" style="width:600px" align="center">
# <p style="font-size: 8pt">Quelle: https://www.sueddeutsche.de/digital/kindesmissbrauch-darknet-kuenstliche-intelligenz-1.4726598</p>

# ### Personality Classification
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/zeitAIbewerbung.PNG" style="width:800px">

# ### Automatic Text Generation
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/textGenerationRetresco.png" style="width:600px">
# 
# 
# Source: [https://www.retresco.de](https://www.retresco.de)

# ### Image Generation and Manipulation
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/jtGan.png" style="width:800px">

# ### Style Transfer
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/styleTransfer.png" style="width:800px">
# 
# Source: [https://deepart.io](https://deepart.io)

# ### Drawing
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/gauGanDemo.png" style="width:800px">
# 
# Source: [http://nvidia-research-mingyuliu.com/gaugan/](http://nvidia-research-mingyuliu.com/gaugan/)

# ## AI Use Cases
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/aiUseCases.png" style="width:800px">
# 
# Source: [https://aai.frb.io/assets/images/Library_Use-Case-Familie.png)

# ## Problems/Challenges of AI
# * Controllability
# * Explainability
# * Reliability
# * Confidence
# * Data Efficiency
# * Common Sense (general intelligence)
# * Causal Reasoning

# <img src="https://maucher.home.hdm-stuttgart.de/Pics/captionbotexample.png" style="width:600px">
# 
# Source: [https://www.captionbot.ai](https://www.captionbot.ai)
# 
