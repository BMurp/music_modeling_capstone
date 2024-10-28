
import pandas as pd
import numpy as np 
import sys
import os
sys.path.insert(0, '../../')
from configuration import DATA_SOURCE_FOLDER, DATA_SOURCE_PATH, MODEL_INPUT_DATA_PATH,FMA_METADATA_PATH,FMA_AUDIO_PATH,GTZAN_METADATA_PATH,GTZAN_AUDIO_PATH, PROJECT_ABSOLUTE_PATH
from library.source_data.data_sources import FreeMusicArchive, GTZAN


class CombinedDataLoader():
    '''Loads each raw data source and provides acccess to unioned result
    Attributes:
        df:  The unioned result of common columns of fma and gtzan datasources  
    
    '''
    def __init__(self):
        self.FMA_METADATA_PATH =FMA_METADATA_PATH
        self.FMA_AUDIO_PATH = FMA_AUDIO_PATH
        self.FMA_AUDIO_ABSOLUTE_PATH =PROJECT_ABSOLUTE_PATH +FMA_AUDIO_PATH
        self.fma = FreeMusicArchive(self.FMA_METADATA_PATH,self.FMA_AUDIO_PATH)
        self.GTZAN_METADATA_PATH = GTZAN_METADATA_PATH
        self.GTZAN_AUDIO_PATH = GTZAN_AUDIO_PATH
        self.GTZAN_AUDIO_ABSOLUTE_PATH =PROJECT_ABSOLUTE_PATH +GTZAN_AUDIO_PATH
        self.gtzan = GTZAN(self.GTZAN_METADATA_PATH,self.GTZAN_AUDIO_PATH)
        self.df = self.get_combined_df()
        self.df_files_available = self.df[self.df.file_available ==1]
        self.df_genres_available = self.df[self.df.label.isnull() ==False]
        self.df_filtered = self.df_files_available[ self.df.label.isnull() ==False ]   
        self.print_data_summary()
        return 
    def print_data_summary(self):
        print('tracks in meta', len(self.df))
        print('tracks with files available in project_data_path: ', len(self.df_files_available))
        print('tracks with top level genres available',len(self.df_genres_available) )
        print('tracks with genres and files (df_filtered)',len(self.df_filtered) )

    
    def get_combined_df(self):
        '''concats fma and gtzan metadata, merges via left join to track ids available in file path 
        this second part ads the file_avalable column which can be used to filter to metadata rows
        where files are available 
        '''
        return pd.merge(
                    pd.concat([data.get_file_meta() for data in [self.fma,self.gtzan]]),
                    self.get_track_id_df_from_available_files(),
                    how= 'left',                 
                    on= 'track_id'                
                )
    
    def get_track_id_df_from_available_files(self):
        '''assembles a data frame of track names as track_ids for each of fma and gtzan
        and returns single dataframe of both unioned togther for allowing knowledge of which files exist 
        '''
        fma_track_ids = self.get_track_ids_from_files(self.FMA_AUDIO_ABSOLUTE_PATH,'fma')
        gtzan_track_ids = self.get_track_ids_from_files(self.GTZAN_AUDIO_ABSOLUTE_PATH,'gtzan')
        return pd.concat([fma_track_ids, gtzan_track_ids])

    def get_track_ids_from_files(self,folder_path, dataset='fma'):
        '''loops through the file path of the provided dataset and returns file availabilty dataframe'''
        
        # List all files in the folder
        track_ids = []
        for root, dirs, files in os.walk(folder_path):
            #print(f"Found directory: {root.split('/')[-1]}", )
            if root.split('/')[-1] != '':
                for file in files:
                    if dataset == 'fma':
                        track_ids.append(file.split('.')[0])
                    else:
                        track_ids.append(file)
                    #print(f" - {file.split('.')[0]}")
                    #print(file)
        track_id_df = pd.DataFrame({'track_id':track_ids, 'file_available': np.ones(len(track_ids))})
        return track_id_df
    
    
class ModelDataLoader():
    '''Loads and provides access to model input data and related information'''
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
                'mfccs_mean',
                'onset',
                'tempo',
                'contrast',
                'tonnetz',
                'mfccs_min',
                'mfccs_max']
        self.add_named_feature_columns()
        
    def add_named_feature_columns(self):
        for index, feature in enumerate(self.feature_names):
            self.df[feature] = self.df.features.map(lambda features: features[index])




#For debug purposes
#print(type(CombinedDataLoader()))
