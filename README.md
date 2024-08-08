# Reinforcement Learning with Mancala Game
In this project I explore the application of reinforcement learning techniques to the two-player turn-based game Mancala. This project is a work in progress, models may be added or improved upon over time.

### Organization
Here is the breakdown of the repo structure:
- [results](/results): Contains all the explanation and analysis of model results. This will contain subdirectories for the various models and training approaches, each of which will contain their own notes.md file journaling my thoughts on their outcomes.
- [code](/code): 
    - [mancala.py](/code/mancala.py): My version of mancala for learning agents to interact with.
    - [play.py](/code/play.py): Game loop for human vs human play through the terminal. This is nothing fancy, basically just for me to debug the game's functionality.
    - [Qnetwork.py](/code/Qnetwork.py): Class for the deep Q network architecture, extended from torch.nn.Module.
    - [agent.py](/code/agent.py): Class for Q-learning agent handling parameter updates (learning), game actions (uses Qnetwork).
    - [train.py](/code/train.py): Training loop for two Q-learning agents to play against each other.
