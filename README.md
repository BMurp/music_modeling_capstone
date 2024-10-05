# music_modeling_capstone
This is the working repository for siads 699 capstone project. This is in the defintion phase now. As project is further defined we can add more details as to the project scope.

# Setup 
1. Install UV following the steps here: https://github.com/astral-sh/uv . This is used for managing python, dependencies, virtual environments, and ipython kernels.
2. Make sure uv is avalable by typing `uv` command in terminal 
3. use terminal to navigate to your chosen local directory for the project 
4. Clone this repo with git clone https://github.com/BMurp/music_modeling_capstone.git
5. `cd music_modeling_capstone` to enter project foler 
6. Install pyton version into environment: `uv python install 3.10`
7. Install project dependencies: `uv sync`
8. Create the virutal environment  `uv env`
9. For Jupiter notebooks, I've tested using Visual Studio Code.  One of the dependencies is iPython, with this the virtual enviroment can be used as a Kernel in visual studio.  For this create a .ipynb file, select uv virtual environment as kernel.  It should be called `.vnv (Python 3.10.5)`.   You can also take a different path like jupyter lab: https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project
