# Q-agent vs Q-agent version 1 training notes
These are the training results of the most preliminary versions of the model. The model architecture at this point is essentially exactly as implemented in this video, just modified to handle the specific application to the game of mancala: https://www.youtube.com/watch?v=wc-FxNENg9U

### Observations
Here are my intial thoughts on the results of the training. These experiments are certainly not exhaustive, but I did not run them longer as they were not seeming to converge to any desirable behavior.
- In each case, one agent seems to always beat the punch to a dominant strategy and completely shut out the other player.
- Sometimes there will be a back and forth of this result, but interestingly the dominance will last for a large number of games, not in brief spurts.

### Shortcomings / routes for improvement
This was the preliminary model, I essentially just wanted a model in which all the pieces actually fit together. That said, I have a number of thoughts of how to move towards something actually effective:
- State representation: Currently, the model takes as input a 14-dim vector that simply expresses exactly the information of the game. I think there is very possibly a more expressive packaging of this information which the model could more quickly learn to use effectively, but this would take considerable thought / research.
- Reward signaling: I suspect that the signaling is much too weak and is possibly responsible for the trends observed in training. Instead of giving nonzero reward only on wins/losses, it might make sense to give smaller signals when marbles are captured mid-game. Another possibility is a premature win reward given when one agent is unable to comeback from the current store deficit on the board. I believe this problem is also connected with the next.
- Training mechanism: For the initial model setup, I basically blindly followed the video's example. One part I did not exactly understand was the batch training, and I am suspect that it does not really accomplish what I would like. Especially given my current reward structure, the training process should be propogating the end reward through the sequence of moves which got the player to a winning position. This will take some more care to implement, but is very doable.

### Alternative models
The current algorithm being employed is Q-learning. This was my first thought for what to be applied to games just by recency bias. Given this is a relatively simple turn based game with finite states (albeit still a fairly large state space), something like a Minimax algorithm might actually be more effective here.