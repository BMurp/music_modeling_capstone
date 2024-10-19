
import pandas as pd
import sys
sys.path.insert(0, '../../')

#print('datasource paths',sys.path)
import fma_modules.utils as fma_utils



class DataSource():
    '''Base class for datasource information
    Attributes:
        metadata_path:  the path to data csv files
        audio_path: the path to audio files
        columns: the defined list of columns to include in dataframe
    
    
    '''
    def __init__(self,metadata_path, audio_path):
        self.metadata_path = metadata_path 
        self.audio_path = audio_path
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
    
    def get_audio_paths(self):
        '''Returns series of audio paths '''
        return

        
class FreeMusicArchive(DataSource):
    '''Specifics of Free Music Archive Data Source'''

    def __init__(self, metadata_path, audio_path):
        DataSource.__init__(self, metadata_path, audio_path)
        self.tracks = tracks =fma_utils.load(self.metadata_path + 'tracks.csv')

    def get_file_meta(self):
        track_meta = self.tracks['track']
        id_and_labels = (track_meta[['genre_top','genres','genres_all']]
                         .rename(columns={'genre_top': 'fma_genre_top',
                                          'genres': 'fma_genres',
                                          'genres_all': 'fma_genres_all'
                                          
                                          })
                         )
        id_and_labels['dataset']= 'fma'
        id_and_labels['audio_path'] = self.get_audio_paths()
        id_and_labels['label'] = id_and_labels['fma_genre_top']
        
        #lower case and replace '-'
        id_and_labels['label'] = id_and_labels['label'].str.lower()
        id_and_labels['label'] = id_and_labels['label'].str.replace('-', '')

        
        return id_and_labels[self.columns]
    
    def get_audio_paths(self):
        return (self.tracks.index
                    .to_series()
                    .map(lambda index: fma_utils.get_audio_path(self.audio_path, index))
                )


class GTZAN(DataSource):
    '''Specifics of GTZAN data source'''

    def __init__(self, metadata_path, audio_path):
        DataSource.__init__(self, metadata_path, audio_path)
        self.features_30_sec = pd.read_csv(metadata_path+ 'features_30_sec.csv')
        return
    def get_file_meta(self):      
        id_and_labels = self.features_30_sec[['filename','label']].reset_index()
        id_and_labels['track_id'] = id_and_labels['filename']

        id_and_labels['dataset']= 'gtzan'

        id_and_labels['audio_path'] = self.audio_path +'/'+ id_and_labels.label + '/' + id_and_labels.filename

        harmonized = id_and_labels.set_index('track_id')
        harmonized['fma_genre_top'] = 'n/a'
        harmonized['fma_genres'] = 'n/a'
        harmonized['fma_genres_all'] = 'n/a'
        
        #consolidate subgenres to match FMA genres 
        harmonized['label'] = harmonized['label'].replace('metal', 'rock')
        harmonized['label'] = harmonized['label'].replace('disco', 'soulrnb')        
        harmonized['label'] = harmonized['label'].replace('reggae', 'international')

        return harmonized[self.columns]
    


#print(type(GTZAN()))