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


### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

the unit test can be run by typing in pytest in the console. The test are checking basic logic.

## Authors

* **Danie Smit** - *Initial work* - [tic-tac-toe and deep learning](https://github.com/D5mit/tic-tac-toe_deep_learning)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Udacity
