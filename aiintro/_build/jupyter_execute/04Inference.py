#!/usr/bin/env python
# coding: utf-8

# # Knowledge-based Systems
# * Author: Johannes Maucher
# * Last update: 26.10.2020

# ![agentKnowledge.png](https://maucher.home.hdm-stuttgart.de/Pics/agentKnowledge.png)

# * Knowledge-based Agents shall provide answers to (complex) questions
# * They are known e.g. as <font color="red"> Expert systems </font>, e.g. for
#     - medical diagnosis
#     - technical diagnosis
# * IBM's <font color = "red">Watson</font> is a famous knowledge based agent
# * Another (fading) popular form of a Knowledge-based Agent is the <font color = "red">Semantic Web</font>
# * More pragmatic:
#     - Google's knowledge graph
#     - DBPedia
#     

# One of the most famous definitions of AI, the **Turing Test** addresses primarily a Knowledge-based agent: 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/turingtest.jpg" style="width:400px" align="center">

# ## General Concept
# 1. <font color="red">Transform</font> unstructured Data into structured form
# 2. Store  <font color="red">knowledge in structured form </font> in database
# 3. Represent  <font color="red">queries</font> in structured form
# 3.  <font color="red">Inference-methods</font> have to generate answers to the queries  

# **Former approaches:**
# * Use propositional logic, first-order-logic or some dialect thereof to formally represent knowledge and infere (PROLOG).
# * Drawbacks:
#     - doesn't scale
#     - logic knows only `True` and `False`
#     
# **More recent approaches:**
# * Knowledge Graphs
# * Bayesian Networks

# ## Expert Systems
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/expertSystem.png" style="width:500px" align="center">

# ## Knowledge Graphs
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/googleKnowledgeGraph.PNG" style="width:600px" align="center">

# * Google Knowledge Graph is the semantic database of Google
# * Used e.g. in the boxes next to the google search results

# **Knowledge Graph:**
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/googleQueryLeonardo.PNG" style="width:600px" align="center">

# **Knowledge Graph:**
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/knowledge-graph.png" style="width:800px" align="center">
# Source: [Watson Discovery Knowledge Graph](https://console.bluemix.net/docs/services/discovery/building-kg.html#watson-discovery-knowledge-graph)

# ## Implementation of Knowledge Graphs
# * Graphs are <font color = "red">sets of triples</font>.
# * Each triple describes a <font color = "red">relation between</font> a <font color = "red">pair of entities</font>.
# * Similar to natural language, where sentences consist of <font color = "red">subject, predicate, object</font>.
# * <font color = "red">Transformation from unstructured to structured data</font>:
#     - NLP (Natural Language Processing) is applied to determine subject,predicate,object
#     - these elements are the triples, which constitute the graph

# ### Knowledge, Graphs, Triple-Stores, Queries
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/wikiEntries.PNG" style="width:600px" align="center">

# **Knowledge, Graphs, Triple-Stores, Queries:**
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/wikiTriples.PNG" style="width:600px" align="center">

# **Knowledge, Graphs, Triple-Stores, Queries:**
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/bandGraph.png" style="width:600px" align="center">

# **Knowledge, Graphs, Triple-Stores, Queries:**
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/wikiQueries.PNG" style="width:600px" align="center">

# ### More complex Problem Solving
# **Prolog Example:** Program consists of facts and rules.
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/prologscreenshot.png" style="width:700px">
# 

# **Solving Logic Puzzles:**
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/cherylsbirthday.jpg" style="width:600px">

# ## Examples of Knowledge Graphs
# 
# ### [DBpedia](http://wiki.dbpedia.org/about)
# * contains English wikipedia in structured form
# * Each entitiy and each relation has a unique URI
# * Knowledge graph is large store of triples
# * Can be queried e.g. using [SPARQL](http://wiki.dbpedia.org/OnlineAccess)

# ### LinkedIn Knowledge Graph
# 
# <img src="https://maucher.home.hdm-stuttgart.de/Pics/knowledgeGraphLinkedInLogo1.jpg" style="width:600px">

# ### Further Examples
# 
# * Facebook Graph API
# * Instagram Graph API
# * Amazon Product Graph
# * AirBnB 
# * IBM Watson
