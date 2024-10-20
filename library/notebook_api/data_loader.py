
import pandas as pd
import sys
sys.path.insert(0, '../../')


#print(sys.path)
from configuration import DATA_SOURCE_FOLDER, DATA_SOURCE_PATH, MODEL_INPUT_DATA_PATH
from library.source_data.data_sources import FreeMusicArchive, GTZAN


class CombinedDataLoader():
    '''Loads each raw data source and provides acccess to unioned result
    Attributes:
        df:  The unioned result of common columns of fma and gtzan datasources  
    
    '''
    def __init__(self):
        self.FMA_MEATADATA_PATH =DATA_SOURCE_PATH +"free_music_archive/fma_metadata/"
        self.FMA_AUDIO_PATH = DATA_SOURCE_FOLDER +"free_music_archive/fma_small/"
        self.fma = FreeMusicArchive(self.FMA_MEATADATA_PATH,self.FMA_AUDIO_PATH)
        self.GTZAN_MEATADATA_PATH = DATA_SOURCE_PATH+"gtzan_dataset/Data/"
        self.GTZAN_AUDIO_PATH = DATA_SOURCE_FOLDER+"gtzan_dataset/Data/genres_original"
        self.gtzan = GTZAN(self.GTZAN_MEATADATA_PATH,self.GTZAN_AUDIO_PATH)
        self.df = self.get_combined_df()
        return 
    def get_combined_df(self):
        
        return pd.concat([data.get_file_meta() for data in [self.fma,self.gtzan]])
    
class ModelDataLoader():
    '''Loads and provides access to model input data'''
    def __init__(self,version = '000'):
        self.df = pd.read_parquet(f'{MODEL_INPUT_DATA_PATH}model_input_{version}')
        self.feature_names = ['spectral_centroids_mean',
                'spectral_centroids_delta_mean',
                'spectral_centroids_accelerate_mean',
                'spectral_bandwidth_mean',
                'spectral_rolloff_mean',
                'zero_crossing_rate_mean',
                'rms_mean',
                'chroma_stft_mean',
                'mfccs_mean']
        self.add_named_feature_columns()
        
    def add_named_feature_columns(self):
        for index, feature in enumerate(self.feature_names):
            self.df[feature] = self.df.features.map(lambda features: features[index])




#For debug purposes
#print(type(CombinedDataLoader()))
