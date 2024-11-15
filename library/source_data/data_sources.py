
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, '../../')

from configuration import PROJECT_ABSOLUTE_PATH,FMA_METADATA_PATH,FMA_SMALL_AUDIO_PATH, FMA_MEDIUM_AUDIO_PATH,FMA_LARGE_AUDIO_PATH,GTZAN_AUDIO_PATH,GTZAN_METADATA_PATH
#print('datasource paths',sys.path)
import fma_modules.utils as fma_utils



class DataSource():
    '''Base class for datasource information
    Attributes:
        metadata_path:  the path to data csv files
        audio_path: the path to audio files
        columns: the defined list of columns to include in dataframe
    
    
    '''
    def __init__(self,metadata_path):
        self.metadata_path = metadata_path 
        self.columns = ['dataset',
                        'audio_path',
                        'label',
                        'fma_genre_top',
                        'fma_genres',
                        'fma_genres_all']
        return 
    
    def get_file_meta(self):
        '''Returns standardized /harmonized dataframe'''


        return 
    def get_audio_folder_path(self):
        '''Returns base path location for directory of audio files'''
        return
    
    def get_audio_absolute_path(self):
        '''Returns absolute path to audio folder '''
        return PROJECT_ABSOLUTE_PATH + self.get_audio_folder_path()
        

    def get_audio_file_paths(self):
        '''Returns series of audio file '''
        return
    
    def get_track_id_from_file_name(self,file_name):
        '''based on provided file name return corresponding track id in metadata'''
        return file_name
    def get_track_ids_from_files(self):
        '''loops through the file path of the provided dataset and returns file availabilty dataframe'''
        
        # List all files in the folder
        track_ids = []
        for root, dirs, files in os.walk(self.get_audio_absolute_path()):
            #print(f"Found directory: {root.split('/')[-1]}", )
            if root.split('/')[-1] != '':
                for file_name in files:
                        track_ids.append(self.get_track_id_from_file_name(file_name))
                
        track_id_df = pd.DataFrame({'track_id':track_ids, 'file_available': np.ones(len(track_ids))})
        return track_id_df

        
class FreeMusicArchive(DataSource):
    '''Specifics of Free Music Archive Data Source'''

    def __init__(self, audio_size = 'small'):
        DataSource.__init__(self, metadata_path = FMA_METADATA_PATH)
        self.tracks =fma_utils.load(self.metadata_path + 'tracks.csv')
        self.audio_size = audio_size

    def get_file_meta(self):
        track_meta = self.tracks['track']
        id_and_labels = (track_meta[['genre_top','genres','genres_all']]
                         .rename(columns={'genre_top': 'fma_genre_top',
                                          'genres': 'fma_genres',
                                          'genres_all': 'fma_genres_all'
                                          
                                          })
                         )
        id_and_labels['dataset']= 'fma'
        id_and_labels['audio_path'] = self.get_audio_file_paths()
        id_and_labels['label'] = id_and_labels['fma_genre_top']
        
        #lower case and replace '-'
        id_and_labels['label'] = id_and_labels['label'].str.lower()
        id_and_labels['label'] = id_and_labels['label'].str.replace('-', '')
        #make track_id same as file
        id_and_labels.index = id_and_labels.index.map(lambda track_id: '{:06d}'.format(track_id))
        #id_and_labels['track_id'] = id_and_labels['track_id'].apply(lambda track_id: '{:03d}'.format(track_id))

        
        return id_and_labels[self.columns]
    
    def get_audio_folder_path(self):
        '''bases path based on audio_size'''
        audio_path = ''
        if self.audio_size == 'small':
            audio_path = FMA_SMALL_AUDIO_PATH
        elif self.audio_size == 'medium':
            audio_path = FMA_MEDIUM_AUDIO_PATH
        elif self.audio_size == 'large':
            audio_path = FMA_LARGE_AUDIO_PATH
        else:
            audio_path = FMA_SMALL_AUDIO_PATH
        return audio_path

    def get_audio_file_paths(self):
        '''returns series of audio paths'''
        return (self.tracks.index
                    .to_series()
                    .map(lambda index: fma_utils.get_audio_path(self.get_audio_folder_path(), index))
                )
    def get_track_id_from_file_name(self,file_name):
        return file_name.split('.')[0]




class GTZAN(DataSource):
    '''Specifics of GTZAN data source'''

    def __init__(self):
        DataSource.__init__(self, metadata_path = GTZAN_METADATA_PATH)
        self.features_30_sec = pd.read_csv(self.metadata_path+ 'features_30_sec.csv')
        return
    def get_file_meta(self):      
        id_and_labels = self.features_30_sec[['filename','label']].reset_index()
        id_and_labels['track_id'] = id_and_labels['filename']

        id_and_labels['dataset']= 'gtzan'

        id_and_labels['audio_path'] = self.get_audio_folder_path() +'/'+ id_and_labels.label + '/' + id_and_labels.filename

        harmonized = id_and_labels.set_index('track_id')
        harmonized['fma_genre_top'] = 'n/a'
        harmonized['fma_genres'] = 'n/a'
        harmonized['fma_genres_all'] = 'n/a'
        
        #consolidate subgenres to match FMA genres 
        harmonized['label'] = harmonized['label'].replace('metal', 'rock')
        harmonized['label'] = harmonized['label'].replace('disco', 'soulrnb')        
        harmonized['label'] = harmonized['label'].replace('reggae', 'international')

        return harmonized[self.columns]
    def get_audio_folder_path(self):
        return GTZAN_AUDIO_PATH

    


#print(type(GTZAN()))