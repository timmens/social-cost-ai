## The Social Costs of Artificial Intelligence

### Overview

Artifical intelligence has grown rapidly over the last decades, followed with a growing demand of computing power.
This massive demand of computing power already makes up a significant portion of emissions for certain industries.
In this project we aim to shed light on the energy consumption of artifical intelligence.

More specifically we plan to do the following:


- *Problem and Algorithm*: Find ML problem related to sustainability on kaggle.com and pick one of the top three approaches.
- *Measurements*: Run algorithm on local hardware to measure actual energy consumption of the system, plus using the method of Strubell for comparison


### Approach to follow on the local machine

I plan to implement a github based project that works with conda.
The approach I'm suggesting requires the server to be connected to the internet, such that it can download the project from github.
It then creates the computing environment using conda. I still have to look into the data management problem. Especially how we will transfer large
amounts of data to the machine. It then runs the algorithm while keeping some kind of logfile on time and internal energy consumption. This logfile
is uploaded when the algorithm is finished.

#### Required software

On the local machine we need (will have to check if this suffices)

- git
- miniconda3
