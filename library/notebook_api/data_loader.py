
import pandas as pd
import sys
sys.path.insert(0, '../../')


#print(sys.path)
from configuration import DATA_SOURCE_PATH
from library.source_data.data_sources import FreeMusicArchive, GTZAN


class CombinedDataLoader():
    '''Loads each data source and provides acccess to unioned result
    Attributes:
        df:  The unioned result of common columns of fma and gtzan datasources
   
    
    
    '''
    def __init__(self):
        self.FMA_MEATADATA_PATH =DATA_SOURCE_PATH +"free_music_archive/fma_metadata/"
        self.FMA_AUDIO_PATH = DATA_SOURCE_PATH +"free_music_archive/fma_small/"
        self.fma = FreeMusicArchive(self.FMA_MEATADATA_PATH,self.FMA_AUDIO_PATH)
        self.GTZAN_MEATADATA_PATH = DATA_SOURCE_PATH+"gtzan_dataset/Data/"
        self.GTZAN_AUDIO_PATH = DATA_SOURCE_PATH+"gtzan_dataset/Data/genres_original"
        self.gtzan = GTZAN(self.GTZAN_MEATADATA_PATH,self.GTZAN_AUDIO_PATH)
        self.df = self.get_combined_df()
        return 
    def get_combined_df(self):
        
        return pd.concat([data.get_file_meta() for data in [self.fma,self.gtzan]])
    

#For debug purposes
#print(type(CombinedDataLoader()))
