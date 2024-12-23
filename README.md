# 'Good Vibrations' Music Modeling Project 
This is the working repository for the Universition of Michigan, Applied Data Science Capstone Project (SIADS699). The Good Vibrations project seeks to fill a critical void in the current music technology landscape by offering a solution for music enthusiasts who manage personal file-based music collections. While streaming services dominate the market, there remains a lack of tools that enable users to organize and interact with their audio files in meaningful ways. This project aims to address this gap by creating a comprehensive framework that leverages audio signal processing and machine learning to classify genres and group music thematically based on sonic characteristics.

## Featured Modeling and Analysis 
The `notebooks/00_featured_notebooks` folder contains the featured analysis and modeling described in the project paper and video.  Exploratory notebooks representing or initial designs and exporations can be found in `notebooks/exploratory`

The following is an itemization of featured notebooks 
- [00_create_modeling_data.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/00_create_modeling_data.ipynb)
- [01_exploratory_label_and_feature_analysis.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/01_exploratory_label_and_feature_analysis.ipynb)
- [02_supervised_learning_explainable_models.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/02_supervised_learning_explainable_models.ipynb)
- [03_deep_learning_model_training.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/03_deep_learning_model_training.ipynb)
- [04_deep_learning_model_evaluation.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/04_deep_learning_model_evaluation.ipynb)
- [05_pca_kmeans_clustering.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/05_pca_kmeans_clustering.ipynb)
- [06_audio_player_demo.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/06_audio_player_demo.ipynb)

## Setup Development Envirionment

### Setup Local Project and Virtual Environment 
1. Install UV following the steps here: https://github.com/astral-sh/uv . This is used for managing python, dependencies, virtual environments, and ipython kernels.
2. Make sure uv is avalable by typing `uv` command in terminal 
3. use terminal to navigate to your chosen local directory for the project 
4. Clone this repo with git clone https://github.com/BMurp/music_modeling_capstone.git
5. `cd music_modeling_capstone` to enter project foler 
6. Install python version into environment: `uv python install 3.10`
7. Install project dependencies: `uv sync`
8. Create the virutal environment  `uv env`
9. For Jupiter notebooks, We've tested using Visual Studio Code.  One of the dependencies is iPython, with this the virtual enviroment can be used as a Kernel in visual studio.  For this create a .ipynb file, select uv virtual environment as kernel.  It should be called `.vnv (Python 3.10.5)`.   You can also take a different path like jupyter lab: https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project

### Setup Instructions For Data Dependencies
To get most notebooks to run, it is required to download the dependent data and host it locally within the designated project folder as outlined in the sections below.  The project includes the original source data, as well as "model input data", which is generated by our feature extraction pipeline and pre-processed for the purpose of modeling. All notebooks rely on specific data files being present in the project folder. As these files are not stored on github, separate steps are needed to download and organize the data. data is stored and distributed using G-Drive.

- For data sufficient to run all notebooks: Download the  "project_data_folder.zip" on UMichigan's g-drive storage [here](https://drive.google.com/file/d/1DqhkK2En67Ebea3bvI0fnT07Ug4xSeqT/view?usp=sharing).  This is 20.5GB

- For the minimum data to run the [06_audio_player_demo.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/06_audio_player_demo.ipynb) notebook with fma_small data download the  "project_data_folder_justdemo.zip" file on UMichigan's g-drive storage [here](https://drive.google.com/file/d/1Pfsw2GFN1rjob3mYh3DUizs5PuAIsBUE/view?usp=sharing).  This is 9GB.

- Once unzipped, make sure the contents of the archive are placed within the projects local data directory `music_modeling_capstone/project_data_folder`. This folder has been added to gitignore to avoid tracking history here.  The specifics of the directory expectations are also confirgurable in `configuration.py`

## About the Data
Desciptions of our source data as well as the data produced by our feature engineering pipelines. 

### Source Data Descriptions
Below is an overview of the source data used for the project.  The `CombinedDataLoader` class defined in the projects library combines the two sources and provides dataframes and methods for interacting with the data. 

#### Free Music Archive
Free Music Archive (FMA) allows for free to use music. For more on this their site here [here](https://freemusicarchive.org/)

For our project we accessed music original sourced from FMA through a publically  available github project: https://github.com/mdeff/fma . The author of this product built an integration with FMA's API and pre-downloaded 100 thousand tracks along with their associtaed metadata. Metadata can be downloaded [here](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip), and is also in our `/project_data_folder/free_music_archive/fma_metdata` . 

The music files themselves are available in mp3 form for download, and there are 3 sizes to pick from, the descriptions of each size are copied below for reference 

- fma_small.zip: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
- fma_medium.zip: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
- fma_large.zip: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
- fma_full.zip: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

For our project, we started designing with fma_small, and then eventually incorporated fma_large.  fma_small is available in the project's g-drive, but fma_large is not given it's size.  It is still hosted and available to download from the github project mentioned above. 

Code from this repo is released under the [MIT License](https://github.com/mdeff/fma/blob/master/LICENSE.txt).  Metadata is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0)

Citation: 
`
@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
`
We do not hold the copyright on the audio and distribute it under the license chosen by the artist.

#### GTZAN Dataset - Music Genre Classification
GTZAN is our second music source for the project, and more details and dowloads can be found [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
This data set is also available in g-drive for the project here `/project_data_folder/gtzan_dataset`

GTZAN is a widely distributed free to use dataset used across many machine learning research papers and Kaggle projects. It is not approved for public distribution for commercial means as we were ubable to locate specific copy right information. 

This dataset was used for the well known paper in genre classification " Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

### Modeling Input Data And Feature Extraction Pipeline
This project introduces a framework for loading and unifying the metadata from the source datasets, extracting numerious features from the associated mp3 files, and writing out datasets of features and corresponding lables to use in modeling and analysis. 

[00_create_modeling_data.ipynb](https://github.com/BMurp/music_modeling_capstone/blob/main/notebooks/00_featured_notebooks/00_create_modeling_data.ipynb) is the notebook that has been used to generate model input data for various scenarios.For Free Music archive data, select functions from  `utils.py` file used for interacting with the source were copied from the source [here](https://github.com/mdeff/fma/blob/master/utils.py) to the projects `fma_modules` directory. 

The outputs of the feature extraction pipeline get stored in the `/project_data_folder/model_input_data` local path and are organized by version and data type.   Note that we utilized a few versions of model input data owing to the fact that we iterated on the data processing pipeline in parallel to iterating on the modeling.  The following is an overview of the versions that may be utilzed through the project. 

1. `model_input_003` : First complete version, includes all numerical features and the full data set including fma_large.  All features generated on full audio durations at full sample rate. This data was used in the initial exploratory analysis and on the supervised learning explainable models section. 
2. `model_input_005` : Also a complete version, introduces the MFCC vector for the first time generated based on full audio at original 44.1KHZ sample rate.  Some analysis is done on this version as we initially thought it would be final version prior to deep learning needs motivating a variant.
3. `model_input_006`: This version is the source for the deep learning components.  It includes both MFCC and Log Melspectrogram vectors, each saved as ndArrays as Numpy files in their own folder, and each generated at 22500 HZ and truncated to 25 seconds. They can be found in the folders labeld `model_input_006_log_melspectrogram`, and `model_input_006_mfcc`. This was also generated after the project team decieded to focus on the six most intuitive genres of rock, electronic, hiphop, classical, jazz, and country.  As such, data was pre-filtered to those generes as part of generating.

The `ModelInputData` class defined in the `data_loader` module is the interface between the notebooks and the data, it reads the data from the file paths and provides access to either dataframes or ndArrays for modeling tasks. 

### Saved Keras Models
The pre-trained models evaluated in the deep learning sections are included as .keras files in the `saved_models` directory of the project.  The deep learning evaluation notebook above loads those models from here and evaluates them using the test data. 

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
  
