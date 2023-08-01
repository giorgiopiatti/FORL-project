# Learning to Bluff and Cooperate: RL in Briscola

Briscola Chiamata, a popular Italian card game, revolves around five players who form unknown partnerships. This game presents a challenge for existing Reinforcement Learning (RL) methods due to its nature of imperfect information, the necessity of effective communication, striking a balance between deceit and sharing information, in order to achieve success. We investigate the capacity of machines to comprehend collaboration and deception in order to enhance their gameplay. Our approach involves the creation of a virtual environment that emulates the game, where we train two deep neural network architectures, specifically an MLP and an RNN through self-play. These agents undergo training against their prior versions, customized heuristic agents, and random agents. To evaluate their performance, we employ a self-developed Elo score. Additionally, we present a comprehensive analysis of the agents' learning progress and the dynamic evolution of bluffing strategies throughout the training process. The findings contribute to understanding the potential of RL agents in mastering complex, multi-agent card games and provide insights into the role of bluffing in strategic gameplay.

#### Authors
Nathan Corecco, David Gu, Giorgio Piatti \
Department of Computer Science, ETH Zurich, Switzerland


## Overview

```
/agents                     -- Heuristic and random agents
/checkpoints                -- Checkpoint of first phase training of communication (only allowed to say truth)
/cleanrl_utils              -- Clean RL utils
/ELO_tournament             -- ELO tournament result analysis
/environtment               
    /briscola_base
        briscola.py         -- Gymnasium environment for the Game of Briscola (Single agent POV) with support for NN architecture only
        briscola_rnn.py     -- Gymnasium environment for the Game of Briscola (Single agent POV) with support for RNN architecture only
        briscola_ELO.py     -- Gymnasium environment for the Game of Briscola (Single agent POV) with support any agent (heuristic, RNN, NN, random)
    /briscola_communication
        briscola.py         -- Gymnasium environment for the Game of Briscola with communication round (Single agent POV) with support for NN architecture only
        briscola_rnn.py     -- Gymnasium environment for the Game of Briscola with communication round (Single agent POV) with support for RNN architecture only
        briscola_ELO.py     -- Gymnasium environment for the Game of Briscola with communication round (Single agent POV) with support any agent (heuristic, RNN, NN, random)
    /briscola_immediate_reward
        briscola.py         -- Gymnasium environment for the Game of Briscola and immediate reward (after each action) (Single agent POV) with support for NN architecture only
/experiments                -- Experiments analysis and plotting for the paper
ELO_parallel.py             -- Parallel ELO computation
ELO.py                      -- Sequential ELO computation
ppo_com_universal.py        -- Train script PPO for universal NN architecture (end result)
ppo_com.py                  -- Train script PPO for single network roles with communication
ppo_RNN_universal.py        -- Train script PPO for universal RNN architecture  (end result)
ppo.py                      -- Train script PPO for single network roles
paper.pdf                   -- paper describing our approaches
```

### Dependencies
This code depends on the following libraries
```
gymnasium                 0.28.1
pettingzoo                1.22.3
wandd                     0.15.0
pytorch                   1.13.1         
numpy                     1.23.4
pandas                    1.5.2 
plotly                    5.12.0
```