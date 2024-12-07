#Defines project level configurations
import os

#make absolute file path of project available
PROJECT_ABSOLUTE_PATH = os.path.abspath(os.path.dirname(__file__))

#Folder For Data to use in relative references
DATA_SOURCE_FOLDER = '/project_data_source/'

#Full path to data within project's absolute path 
DATA_SOURCE_PATH = PROJECT_ABSOLUTE_PATH + DATA_SOURCE_FOLDER

#Expected location of fms metadata file csvs
FMA_METADATA_PATH = DATA_SOURCE_PATH +"free_music_archive/fma_metadata/"

#relative path to the FMA audio
#this gets printed to data sources, extraction process adds absolute path
FMA_SMALL_AUDIO_PATH = DATA_SOURCE_FOLDER +"free_music_archive/fma_small/"
FMA_MEDIUM_AUDIO_PATH = DATA_SOURCE_FOLDER +"free_music_archive/fma_medium/"
FMA_LARGE_AUDIO_PATH = DATA_SOURCE_FOLDER +"free_music_archive/fma_large/"

#Expected location of gtzan metadadata files 
GTZAN_METADATA_PATH = DATA_SOURCE_PATH+"gtzan_dataset/Data/"

#Gtzan relative path to audio files 
GTZAN_AUDIO_PATH = DATA_SOURCE_FOLDER+"gtzan_dataset/Data/genres_original"

#Model Input Data 
MODEL_INPUT_DATA_FOLDER = 'model_input_data/'
MODEL_INPUT_DATA_PATH = DATA_SOURCE_PATH + MODEL_INPUT_DATA_FOLDER

#Saved Model Path 
SAVED_MODEL_PATH = PROJECT_ABSOLUTE_PATH + '/saved_models/'