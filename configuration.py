import os

#Define project level configurations

#make absolute file path of project available
PROJECT_ABSOLUTE_PATH = os.path.abspath(os.path.dirname(__file__))


DATA_SOURCE_FOLDER = '/project_data_source/'
DATA_SOURCE_PATH = PROJECT_ABSOLUTE_PATH + DATA_SOURCE_FOLDER


MODEL_INPUT_DATA_FOLER = '/model_input_data/'
MODEL_INPUT_DATA_PATH = DATA_SOURCE_PATH + MODEL_INPUT_DATA_FOLER