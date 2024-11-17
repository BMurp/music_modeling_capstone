
import pandas as pd
import numpy as np 
import sys
import os
import glob
sys.path.insert(0, '../../')
from configuration import  MODEL_INPUT_DATA_PATH
from library.source_data.data_sources import FreeMusicArchive, GTZAN


class CombinedDataLoader():
    '''Loads each raw data source and provides acccess to unioned result
    Attributes:
        df:  The unioned result of common columns of fma and gtzan datasources  
    
    '''
    def __init__(self, fma_audio_size = 'small'):
        self.fma = FreeMusicArchive(fma_audio_size)
        self.gtzan = GTZAN()
        self.df = self.get_combined_df()
        self.df_files_available = self.df[self.df.file_available ==1]
        self.df_genres_available = self.df[self.df.label.isnull() ==False]
        self.df_filtered = self.df_files_available[self.df.label.isnull() ==False ]   
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
                    pd.concat([data.get_track_ids_from_files() for data in [self.fma,self.gtzan]]),
                    how= 'left',                 
                    on= 'track_id'                
                )

    
    
class ModelDataLoader():
    '''Loads and provides access to model input data and related information'''
    def __init__(self,version = '000'):
        self.version = version
        self.data_path = f'{MODEL_INPUT_DATA_PATH}model_input_{self.version}'
        self.df = self.get_df_from_parquet()
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
        self.label_names = self.df.label.unique()
        self.class_distribution = pd.DataFrame(self.df['label'].value_counts(normalize=True) * 100).reset_index()

        self.add_named_feature_columns()
    
    def test_row_uniqueness(self,model_data_loader_df):
        '''asserts that there are no duplicate rows in provided dataframe
        if a file appears on more then one row - or if indexes are not unique
        it can lead to unpredictable results in filtering and splitting 
        '''
        try:
            assert len(
                set(
                    (len(model_data_loader_df),
                    len(pd.Series(model_data_loader_df.index).value_counts()),
                    len(model_data_loader_df.groupby('track_id').count()),
                    len(model_data_loader_df.groupby('audio_path').count())
                    )
                )
            ) == 1, 'ERROR: Incorrect Row-Wise Structure of DataFrame, there are duplicate indexes or audio files'
        except AssertionError as msg:
            print(msg)
    def get_df_from_parquet(self):
        '''reads data frame from parquet, drops unneeded columns resets index'''
        df = pd.read_parquet(self.data_path)
        #drop not needed columns, while checking if column exists for backward compatibility with different data versions
        drop_columns = [column for column in df.columns if column in ['level_0','index']]
        df.drop(columns=drop_columns)
        #reset the index to 0 based serial
        df.reset_index(inplace=True)
        #row uniques test
        self.test_row_uniqueness(df)
        return df
    
    def add_named_feature_columns(self):
        for index, feature in enumerate(self.feature_names):
            self.df[feature] = self.df.features.map(lambda features: features[index] if features is not None else None)

    def get_mfcc(self):
        npy_path = self.data_path + '_mfcc/*npy'
        files = glob.glob(npy_path)
        mfcc_array = []
        for file in files:
            mfcc_array.append(np.load(file, allow_pickle=True))
        #mfcc_array
        combined_array = np.concatenate(mfcc_array, axis=0)

        return combined_array







#For debug purposes
#print(type(CombinedDataLoader()))
