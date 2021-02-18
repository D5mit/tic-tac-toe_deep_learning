# Tic Tac Toe and Deep Learning

The objective of this notebook is to train an AI agent to play Tic Tac Toe. The AI Agent knows nothing about Tic Tac Toe, it does not know any of the rules of the game or know that it is even playing a game. The AI agent will play multiple games of Tic Tac Toe and will learn from it. The Tac Tac Toe game will allow the Agent to play a move on one of the 9 positions on a Tic Tac Toe board. The game will then check if this position is already filled (therefore invalid move) and if filled, will ask for a new move to be played. If the position is not already filled, the game will check if this leads to a win, if the game has been won, the game gives feedback that it was won.

## Getting Started

The notebook <strong>tic_tac_toe_dl.ipynb</strong> contain the training logic. The program has wo main processes, the generation of data and the training of the models. 
The parameters below in <strong>tic_tac_toe_dl.ipynb</strong> as defined in under <strong>Parameters</strong>, manages the parts that will be executed.  They can be changed as required.

Setting all these parameters to True in the section below, will lead to the notebook running rather long.
do_gen_ifile_UU -> set to true if untrained agent should play against untrained agent
do_gen_ifile_US -> set to true if untrained agent should play against untrained agent
do_gen_ifile_SS -> set to true if untrained agent should play against agent S
do_gen_ifile_SH -> set to True if agent S should play against agent S
do_train_modelS -> set to True if agent s vs human game should be played
do_train_modelL -> set to True if model S should be trained
do_final_evaluL -> set to True if the final evaluation should be executed


### Overview of files:
tic_tac_toe_dl.ipynb    - Jupyter notebook containing project to train an AI agent to play Tic Tac Toe
ttt.py                  - python script will the Tic Tac Toe game logic
main.py                 - Program to be able to run and test the game scripts
README.md               - Explanation of project
modelAgentLinki.h5      -   
modelAgentLinki.json    -  
modelAgentSmit.h5       -
modelAgentSmit.json     -





### Installing

Packages required:
```
pip3 install pytest
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install keras
pip3 install tensorflow
pip3 install sklearn
```

## Training data
The game creates its own training data, however this takes a while to run, for each of use some training data is included:
[Training data](https://s3.us-east-2.amazonaws.com/d5mit.co.za/ttt_training_data/training_data.zip)
The notebook uses this data, however this can be changed in the Parameters section.

## Running the tests

the unit test can be run by typing in pytest in the console. The test are checking basic logic.

## See the model in action
To see the model in action you can visit these two apps:
- [Tic Tac Toe](http://ec2-3-17-207-44.us-east-2.compute.amazonaws.com/tictactoe)
- [Game of scrabble](http://ec2-3-17-207-44.us-east-2.compute.amazonaws.com/tictactoecard) <br>
  (github link for these applications: [d5mit](https://github.com/D5mit/d5mit_flask))

## Author

* **Danie Smit** - *Initial work* - [tic-tac-toe and deep learning](https://github.com/D5mit/tic-tac-toe_deep_learning)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Udacity
