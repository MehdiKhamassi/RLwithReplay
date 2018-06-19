# RLwithReplay
This is the source code to simulate model-based (MB) and model-free (MF) reinforcement learning algorithms with replays in grid worlds.

i.e., the code simulates either reactivations of episodic memory buffer during learning phase for MF algorithms, or mental simulations of (state,action,new_state,reward) quadruplet events with the internal model during inference phase for MB algorithms.

The code is currently adapted to simulate reinforcement learning models in a discretize world representing the multiple-T-maze experiment of A. David Redish and colleagues and implemented as a Markov Decision Process.

This code goes with the following submission: Cazé*, Khamassi*, Aubin, Girard, Hippocampal replays under the scrutiny of reinforcement learning models. (* equally contributing authors) submitted to Journal of Neurophysiology (special issue "Where Are You Going? The Neurobiology of Navigation »)

## Questions?

Contact Mehdi Khamassi (firstname (dot) lastname (at) upmc (dot) fr)

## Quick start

Use main.m to launch a simulation experiment and to plot a few figures.
If you want to change the model, change the variable called ‘replayMethod’ at the top of main.m
There you can also change the duration of the experiment, the duration of each condition (possible conditions: reward on the left side of the maze; reward on the right side), the departure state, states where replays are authorized, and some other constraints of the task.

After launching a simulation experiment, you can save the corresponding data in a .mat file. Ex: save([‘MyDirectory/MyModel_Expe1.mat’]).

Repeat the same process to save at least 10 experiments for a given model.

After doing this, you can plot article figures using the script called plotFigure10experimentsPerAlgo.m. At the top of this script, first indicate the name of the tested model, the name of your directory, as well as a few task parameters.

Now you can play with the code to compare different models. You can also modify the parameters of the model in function replayAgent.m. You can also modify the maze and parameters of the task in function MDP.m.
