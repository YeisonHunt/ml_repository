# ML-Handover
## Readme
These are the instructions to install and execute the different Reinforncenment
Learning Algorithms in your local machine

## Installation
All the steps are done in a Windows 10 system with Anaconda for the management
of the virtual environments

### Step 1.
Be sure you have Python 3 installed

### Step 2.
Install Anaconda. 
Follow the [documentation](https://docs.anaconda.com/anaconda/install/index.html) 

### Step 3.

#### Option 1
Create a python3 environment with anaconda. You can do it with the UI or with the terminal. Be sure to specify the python 3.7 version.
#### Option 2

Create a python3 environment inside your project folder
```
python3 -m venv /path/to/new/virtual/environment
```
Follow the [documentation](https://docs.python.org/3/library/venv.html)


### Step 4.

Activate the environment.

### Step 5

Execute the requirement folder and install all the dependencies.

```
pip install -r requirements.txt
```
The requirement file is inside the folder.
This will install all the python libraries for the project, gym, tensorflow, scipy ant others.

### Step 6

Execute the different agents

#### Step 6.1 taxi_gym with QLearning

```
cd examples/
python taxi_gym.py
```

#### Step 6.2 egreedy agent

```
cd agents/
python rl_egreedy.py
```
#### Step 6.3 dql agent with gym env

```
cd agents/gym_slicing
python main.py
```

#### Step 6.4 Network slice selection with QLearning and networkx

```
cd agents/gym_slicing
python slicing_main.py
```

### Step 7 New environment definition
Inside the final folder we have the latest environment and agents definition. Inside the folder envs we have the environment definition called "gym-handover". And in the root
we have 3 agents that are: Q-Learning, SARSA, and Deep Q-Learning. Also we use another two agents from the library stable-baselines that are: DQN and A2C. It is important to said that DQL and DQN are not working yet.

For the execution of the training agents we need to make the next steps:

#### 7.1 Install gym-handover environment with pip
```
cd final/envs
pip install -e gym-handover
```
Just in case you have some error, execute:
```
pip uninstall gym-handover
```
### 7.2 Train the agents
``` 
cd final/
python final_exec.py
```
### 8 Obtained Results

Reward values
QLearning
total time: 521.6279995441437
Mean reward QLearning Agent(norm): -7346.784056122449 after convergence: -7321.172
SARSA
total time: 1206.7185084819794
Mean reward Sarsa Agent (norm): -7360.00918367347 after convergence: -7335
A2C
Train Mean Reward A2C_norm: -7520.2055560539629

al time: 485.397216796875
Mean reward QLearning Agent(norm): -7344.924438775511 after convergence: -7311.295
SarsaAgent
total time: 1209.4093935489655
Mean reward Sarsa Agent (norm): -7356.640586734694 after convergence: -7322
