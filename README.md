# 'Good Vibrations' Music Modeling Project 
This is the working repository for siads 699 capstone project. This is in the development phase now. The project seeks to define the foundational elements for a product geared at helping music enthusiasts organize their file based music collections. 

Ths project includes an an end-to-end machine learning pipeline including feature extraction from audio files, an in-depth exploration of music genre classification models, as well as an unsupervised learning component geared at defining custom thematic groupings based on the sonic qualities of the music files. 

## Setup Development Envirionment
1. Install UV following the steps here: https://github.com/astral-sh/uv . This is used for managing python, dependencies, virtual environments, and ipython kernels.
2. Make sure uv is avalable by typing `uv` command in terminal 
3. use terminal to navigate to your chosen local directory for the project 
4. Clone this repo with git clone https://github.com/BMurp/music_modeling_capstone.git
5. `cd music_modeling_capstone` to enter project foler 
6. Install python version into environment: `uv python install 3.10`
7. Install project dependencies: `uv sync`
8. Create the virutal environment  `uv env`
9. For Jupiter notebooks, I've tested using Visual Studio Code.  One of the dependencies is iPython, with this the virtual enviroment can be used as a Kernel in visual studio.  For this create a .ipynb file, select uv virtual environment as kernel.  It should be called `.vnv (Python 3.10.5)`.   You can also take a different path like jupyter lab: https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project

## Development Workflow 
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

## About the Data
### Data Storage Strategy
data is stored and distributed using G-Drive to avoid limitations of github.   There is a "project_data_folder" on UMichigan's g-drive storage [here](https://drive.google.com/drive/u/0/folders/1iEgWbgOzuWd41frPpWAAUADBUJnJGC0p), and the expectation is this folder and it's structure is replicated locally within `music_modeling_capstone/project_data_folder`. This folder has been added to gitignore to avoid tracking history here. 

### Source Data Descriptions
#### Free Music Archive
Free Music Archive (FMA) allows for free to use music. For more on this their site here [here](https://freemusicarchive.org/)

For our project we accessed music original sourced from FMA through a publically  available github project: https://github.com/mdeff/fma . The author of this product built an integration with FMA's API and pre-downloaded 100 thousand tracks along with their associtaed metadata. Metadata can be downloaded [here](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip), and is also in our `/project_data_folder/free_music_archive/fma_metdata` . 

The music files themselves are available in mp3 form for download, and there are 3 sizes to pick from, the descriptions of each size are copied below for reference 

- fma_small.zip: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
- fma_medium.zip: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
- fma_large.zip: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
- fma_full.zip: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

For our project, we started designing with fma_small, and then eventually incorporated fma_large.  fma_small is available in the project's g-drive, but fma_large is not given it's size.  It is still hosted and available to download from the github project mentioned above. 

#### GTZAN Dataset - Music Genre Classification
GTZAN is our second music source for the project, and more details and dowloads can be found [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
This data set is also available in g-drive for the project here `/project_data_folder/gtzan_dataset`

### Feature Extraction Pipeline 
This project introduces a framework for loading and unifying the metadata from the source datasets, extracting numerious features from the associated mp3 files, and writing out datasets of features and corresponding lables to use in modeling and analysis. 

For Free Music archive data, select functions from  `utils.py` file used for interacting with the source were copied from the source [here](https://github.com/mdeff/fma/blob/master/utils.py) to the projects `fma_modules` directory. 

[create_modeling_data.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/data_creation/create_modeling_data.ipynb) is the notebook that has been used to generate model input data for various scenarios. 

## Modeling and Analysis 
The modeling and analysis is ongoing and currently across several notebok in the `/notebooks/exploratory` project folder.

Some good representative notebooks are mentioned and linked below:

- [supervised_learning_explainable_models.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/exploratory/supervised_learning_explainable_models.ipynb)
- [CNN_LSTM_MFCC_Classification.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/exploratory/CNN_LSTM_MFCC_Classification.ipynb)
- [pca_k_means_clustering.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/exploratory/pca_k_means_clustering.ipynb)
- [audio_explorer_design.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/exploratory/audio_explorer_design.ipynb)