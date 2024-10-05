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

# Development Workflow 
A Continuous Integration workflow featuring relatively frequent branches and pull request is proposed so as to all be able to commit our own work while getting a chance to review and build off eachothers work. 

1. create a branch: https://www.git-tower.com/learn/git/faq/create-branch
2. checkout the branch `git checkout <branch_name>`
3. every update can be committed to the branch
4. when ready to share create pull request of the feature branch to main
5. share pull request link in chat
6. To discuss workflows on reviews/ approvals -- if work is not conflicting with others probably fine to merge,  if there are conflicts, should discuss 