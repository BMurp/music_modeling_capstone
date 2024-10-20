#import sys
#sys.path.insert(0, '../../')
import librosa
import numpy as np 
import sys
sys.path.insert(0, '../../')
from configuration import PROJECT_ABSOLUTE_PATH, MODEL_INPUT_DATA_PATH





class AudioFeatureExtractor():
    '''interface for audio feature extraction libraries
    takes a dataframe of the source data as input
    provides method for adding extracted audio data and features to dataframe

    Attributes:
        source_data: should be based on CombinedDataLoader.df, a subset of rows can be passed for testing
        df : current state of the data frame 
     
    '''
    def __init__(self,source_data):
        self.df = source_data.copy()
        return
    
    def get_audio_data(self,file_name):
        try:
            #print('Processing File ', file_name)
            #print('Run Librosa Load')
            y, sr = librosa.load(PROJECT_ABSOLUTE_PATH+file_name, sr=None)
            #print('extracting_features')
            feature_values = np.array(list(self.extract_features(y,sr).values())).astype('float32')
        except:
            return None
        return None, sr, feature_values #do not return full audio
        return y, sr, feature_values #for returning full audio data as well
      
    
    def add_audio_data_to_df(self):
        self.df['audio_data'] = self.df['audio_path'].apply(self.get_audio_data)
        print('putting features to their own columns')
        #self.df['librosa_load'] = self.df['audio_data'].apply(lambda data: data[0] if data is not None else None)
        self.df['sampling_rate'] = self.df['audio_data'].apply(lambda data: data[1] if data is not None else None)
        self.df['features'] = self.df['audio_data'].apply(lambda data: data[2] if data is not None else None)


        return

    def save_results(self, version= '000'):
         #just save data where audio was succesfully returned
         to_save = self.df[self.df['audio_data'].isnull() == False]
         #force object type colums to strings to avoid errors
         to_save['fma_genres'] = to_save['fma_genres'].astype('string')
         to_save['fma_genres_all'] = to_save['fma_genres'].astype('string')
         #put index as a column as well and set to string
         to_save.reset_index(inplace=True)
         to_save['track_id'] = to_save['track_id'].astype('string')
         
         #drop audio_data and save to parquest
         to_save.drop(columns=['audio_data']).to_parquet(f'{MODEL_INPUT_DATA_PATH}model_input_{version}',index=True)
    
    def extract_features(self,y,sr):
        try:
            #y, sr = librosa.load(file_path, sr=None)

            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroids_mean = np.mean(spectral_centroids)
            spectral_centroids_delta_mean = np.mean(librosa.feature.delta(spectral_centroids))
            spectral_centroids_accelerate_mean = np.mean(librosa.feature.delta(spectral_centroids, order=2))

            # Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)

            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff)

            # Zero Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)

            # RMS (Root Mean Square) Energy
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)

            # Chroma STFT
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_stft_mean = np.mean(chroma_stft)

            # MFCCs (Mel-Frequency Cepstral Coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1).mean()

            # Return all features as a dictionary
            return {
                'spectral_centroids_mean': spectral_centroids_mean,
                'spectral_centroids_delta_mean': spectral_centroids_delta_mean,
                'spectral_centroids_accelerate_mean': spectral_centroids_accelerate_mean,
                'spectral_bandwidth_mean': spectral_bandwidth_mean,
                'spectral_rolloff_mean': spectral_rolloff_mean,
                'zero_crossing_rate_mean': zero_crossing_rate_mean,
                'rms_mean': rms_mean,
                'chroma_stft_mean': chroma_stft_mean,
                'mfccs_mean': mfccs_mean
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None  # Return None if there’s an error