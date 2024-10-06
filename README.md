# music_modeling_capstone
This is the working repository for siads 699 capstone project. This is in the defintion phase now. As project is further defined we can add more details as to the project scope.

# Setup Developement Envirionment
1. Install UV following the steps here: https://github.com/astral-sh/uv . This is used for managing python, dependencies, virtual environments, and ipython kernels.
2. Make sure uv is avalable by typing `uv` command in terminal 
3. use terminal to navigate to your chosen local directory for the project 
4. Clone this repo with git clone https://github.com/BMurp/music_modeling_capstone.git
5. `cd music_modeling_capstone` to enter project foler 
6. Install python version into environment: `uv python install 3.10`
7. Install project dependencies: `uv sync`
8. Create the virutal environment  `uv env`
9. For Jupiter notebooks, I've tested using Visual Studio Code.  One of the dependencies is iPython, with this the virtual enviroment can be used as a Kernel in visual studio.  For this create a .ipynb file, select uv virtual environment as kernel.  It should be called `.vnv (Python 3.10.5)`.   You can also take a different path like jupyter lab: https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project

# Managing Data
data is stored and distributed using G-Drive.  There is a "project_data_folder" on g-drive, and the expectation is this folder and it's structure is replicated locally within `music_modeling_capstone/project_data_folder`.  This folder has been added to gitignore to avoid tracking history here 

For intial setup
1. Go to the project_data_source folder here: https://drive.google.com/drive/u/1/folders/1iEgWbgOzuWd41frPpWAAUADBUJnJGC0p
2. Download contents of folder and put in `music_modeling_capstone/project_data_folder`, while ensuring your local copy is the same as g-drive copy including folder names and structure. 
3. Data updates can be communicated in PRs to facilitate a manual process of retrieving data updates along with merging code updates. 

For Free Music Archive Data
For utilizing the pre-pepped data from here: https://nbviewer.org/github/mdeff/fma/blob/outputs/usage.ipynb 
the utils.py file was copied to  `fma_modules` directory.  Currently only the load function is used so all others are commented out.

# Development Workflow 
A Continuous Integration workflow featuring relatively frequent branches and pull request is proposed so as to all be able to commit our own work while getting a chance to review and build off eachothers work. 

1. create a branch: https://www.git-tower.com/learn/git/faq/create-branch
2. checkout the branch `git checkout <branch_name>`
3. every update can be committed to the branch
4. push updates to remote periodically - before that you need to set the upstreamorgin  `git push --set-upstream origin <branch_name>`
5. when ready to share create pull request of the feature branch to main
6. If code update involed new data, add data to the project_data_folder on g-drive and mention this in the PR. 
7. New dependencies can be added with `uv add <libraryname>`, the uv related files such as uv.lock and pyproject.toml will then be updated, these changes should be committed to the branch. 
8. share pull request link in chat
9. To discuss workflows on reviews/ approvals -- if work is not conflicting with others probably fine to merge,  if there are conflicts, should discuss 
