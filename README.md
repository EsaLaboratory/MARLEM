# Multi Agent Reinforcement Learning for Local Energy Markets (MARLEM)

Overview
=============

Energy System Architecture Lab ESAL's [OPLEM](https://github.com/EsaLaboratory/OPLEM) and Autonomous Agents Research Group's [epyMARL](https://github.com/uoe-agents/epymarl) were interfaced to create MARLEM: a multi-agent reinforcement learning (MARL) tool for training and testing reinforcement learning methods on local energy markets (LEM) designs.

The fundamental structure of an RL system consists of an environment that interacts with an agent system through signals: states, actions and rewards. The schema below shows the general structure of RL:
![RL_schema](https://github.com/EsaLaboratory/MARLEM/assets/65967906/49608a50-c0d4-495d-ac10-c22eebf2c0fd|width=75px])

MARL extends the concept of RL and includes multiple agents instead of one:
![MARL_schema](https://github.com/EsaLaboratory/MARLEM/assets/65967906/70691924-c3d1-4b05-893d-ba3fff30ae02|width=75px])

In MARLEM, OPLEM plays the role of the environment and epyMARL the agent(s):
![MARLEM_schema](https://github.com/EsaLaboratory/MARLEM/assets/65967906/ce72c5e2-9039-4e12-a055-172793f6ea09|width=75px])

Documentation
-------------
OPLEM documentation can be found [here](https://open-new.readthedocs.io/en/latest/), and epyMARL documentation [here](https://agents.inf.ed.ac.uk/blog/epymarl/)

Requirements
------------
EpyMARL is more efficient in Linux

Installation
-------------
1. Create a conda virtual environment:
```
conda create --name <name_env> python
```
and activate it: `conda activate <name_env>`

2. install oplem package and all the epyMARL package dependencies by running the following 

```
pip install git+https://github.com/EsaLaboratory/MARLEM.git
```
3. Locate to the directory which will host the epymarl codebase and clone it:
```
git clone https://github.com/uoe-agents/epymarl.git
```
4. and move to the subfolder epymarl: `cd epymarl/`

Getting started
----------------

We have developed an environment in OPLEM, which is a reduced european low voltage network (EULV) with 55 buses connecting inflexible loads and  solar photovoltaic (PV) panels. In the periods of peaks, if the PVs are not well monitored, their export may lead to network violations, particularly voltage violations.
- Agents: PV panels
- Observation: 
- Actions: Amount of power to be curtailed
- Reward: the total revenue - the total violations

To train this environment, you have first to register it with gym.
1. In the gym folder under your virtual environment: `\[path to Anaconda\]\Anaconda\envs\\[your_env\]\Lib\site-packages\gym\envs`
2. Copy the following into the `__init__.py` file:
```
register(
    id="DiscLfmAggPEulv-v0",
    entry_point="oplem:DiscLfmLvAggPEnv",
    kwargs={'network_type':'eulv_reduced'},
)
```

Then run the training: 
3. change directory to where epyMARL was installed
4. activate the virtual environment that contains the MARLEM packages
5. run the following command: 
```
python src/main.py --config=mappo --env-config=gymma with env_args.time_limit=50 env_args.key="oplem.DiscLfmAggPEulv-v0"
```

Advanced Applications
---------------------
For more advanced usage, create your own environment under the oplem package: `\[path to Anaconda\]\Anaconda\envs\\[your_env\]\Lib\site-packages\oplem`

The environment should be gym compatible, i.e., contains the following methods:
- reset()
- step()
And registered in Gym following similar template as in step.2 in **Getting started** section

Contributors
------------
- ESAL group, School of Engineering, University of Edinburgh
- AAR group, School of Informatics, University of Edinburgh



