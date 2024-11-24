#import sys
#sys.path.insert(0, '../../')
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np 
import sys
import os
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
    def __init__(self,source_data,sample_rate = None, start_sample=0, end_sample=None):
        self.sample_rate = sample_rate
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.df = source_data.copy()
        return
    
    def run_extraction_thread(self, version, batch, thread):
        '''for combining extraction and saving for the purpose of parallelization
        puts each result into it's own file 
        '''
        self.add_audio_data_to_df()
        self.add_numerical_features_to_df()
        self.add_mfcc_to_df()
        self.add_log_melspectrogram_to_df()
        self.save_results(version, batch, thread)


    def logger_function(self,thread, start_record, end_record):
        print(f"starting thread {thread} for data between index {start_record} and {end_record}")
    

    
    def add_audio_data_to_df(self,sr=None,start_sample=0,end_sample=None):
        self.df['audio_and_sampling_rate'] = self.df['audio_path'].apply(self.get_audio_and_sampling_rate)
        self.df['audio'] = self.df['audio_and_sampling_rate'].apply(lambda data: data[0] if data is not None else None)
        self.df['sampling_rate'] = self.df['audio_and_sampling_rate'].apply(lambda data: data[1] if data is not None else None)
        return
    
    def add_numerical_features_to_df(self):
        self.df['features'] = self.df['audio_and_sampling_rate'].apply(self.extract_numerical_features)
        return
    
    def add_mfcc_to_df(self):
        self.df['mfcc'] = self.df['audio_and_sampling_rate'].apply(self.get_mfcc)
        return
    
    def add_log_melspectrogram_to_df(self):
        self.df['log_melspectrogram'] = self.df['audio_and_sampling_rate'].apply(self.get_log_melspectrogram)
        return
    
    def get_audio_and_sampling_rate(self,file_name):
        try:
            y, sr = librosa.load(PROJECT_ABSOLUTE_PATH+file_name)
        except:
            print('failure in librosa.load')
            return None

        if self.end_sample is None:
            return y,sr
        else:
            return y[self.start_sample:self.end_sample], sr
    
    def get_mfcc(self, audio_and_sampling_rate):
        if audio_and_sampling_rate is None:
            return None
        y, sr = audio_and_sampling_rate[0],audio_and_sampling_rate[1]
        
        #referencing the power_to_db approach from https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification'''
        #mel spectrogram stub code
        #mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        #mels_db = librosa.power_to_db(S=mels, ref=1.0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc
    
    def get_log_melspectrogram(self, audio_and_sampling_rate):
        if audio_and_sampling_rate is None:
            return None
        y, sr = audio_and_sampling_rate[0],audio_and_sampling_rate[1]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S

  
    def extract_numerical_features(self,audio_and_sampling_rate):
        try:
            if audio_and_sampling_rate is None:
                return None
            y, sr = audio_and_sampling_rate[0],audio_and_sampling_rate[1]

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
            mfccs_min = np.min(mfccs, axis=1).min()
            mfccs_max = np.max(mfccs, axis=1).max()
            
            #Onset and Tempo 
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_env_mean = np.mean(onset_env)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            
            #Contrast 
            
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast )

            #Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            tonnetz_mean = np.mean(tonnetz)
    
            # Return all features as a dictionary
            return np.array([
                spectral_centroids_mean,
                spectral_centroids_delta_mean,
                spectral_centroids_accelerate_mean,
                spectral_bandwidth_mean,
                spectral_rolloff_mean,
                zero_crossing_rate_mean,
                rms_mean,
                chroma_stft_mean,
                mfccs_mean,
                onset_env_mean,
                tempo,
                contrast_mean,
                tonnetz_mean,
                mfccs_min,
                mfccs_max
            ]).astype('float32')
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None  # Return None if there’s an error
        
    def save_results(self, version= '000', batch = None, thread= None):
         #just save data where audio was succesfully returned
         to_save = self.df[self.df['audio_and_sampling_rate'].isnull() == False]
         #force object type colums to strings to avoid errors
         to_save['fma_genres'] = to_save['fma_genres'].astype('string')
         to_save['fma_genres_all'] = to_save['fma_genres'].astype('string')
         #put index as a column as well and set to string
         to_save.reset_index(inplace=True)
         to_save['track_id'] = to_save['track_id'].astype('string')
         
         #drop audio_data and save to parquest
         #run not run by parallel processor write to one file in model_input directory
         to_save = to_save.drop(columns=['audio_and_sampling_rate','audio'])

         version_file_location = f'{MODEL_INPUT_DATA_PATH}model_input_{version}'
         

         if thread is None:
            to_save.to_parquet(version_file_location,index=True)
         #When running in parallel processor mode write files within a versioned path with batch and thread in filename
         else:
            #make the folder for this version if not exist
            os.makedirs(f"{version_file_location}/", exist_ok=True) 
            #write the file to folder
            batch_string = '{:03d}'.format(batch)
            thread_string = '{:03d}'.format(thread)
            os.makedirs(f"{version_file_location}_mfcc/", exist_ok=True) 
            np.save(f"{version_file_location}_mfcc/{batch_string}_{thread_string}.npy", np.array(to_save['mfcc']))
            os.makedirs(f"{version_file_location}_log_melspectrogram/", exist_ok=True) 
            np.save(f"{version_file_location}_log_melspectrogram/{batch_string}_{thread_string}.npy", np.array(to_save['log_melspectrogram']))
            to_save.drop(columns=['mfcc','log_melspectrogram']).to_parquet(f"{version_file_location}/{batch_string}_{thread_string}",index=True)
    
    def explore_features(self, limit=5):
        df = self.df.head(limit)
        indexes = list(range(0,len(df)))
        audio_players = []
        fig, ax = plt.subplots(nrows=len(df), ncols=3, sharex=False)

        for i in indexes:
            row_num = i
            track_id_ =df['track_id'].iloc[row_num]
            mfccs_ = df['mfcc'].iloc[row_num]#[0: int(len(mfccs)/4)]
            log_s_ =df['log_melspectrogram'].iloc[row_num]
            audio_ = df['audio'].iloc[row_num]
            sr_ = df['sampling_rate'].iloc[row_num]
            label_ = df['label'].iloc[row_num]
            print("Showing Track: ", track_id_, "label is: ", label_)
            audio_players.append(ipd.Audio(audio_, rate=sr_))
            '''
            librosa.display.waveshow(audio_, sr=sr_)
            fig, ax = plt.subplots(nrows=2, sharex=True)
            
            img = librosa.display.specshow(log_s_,
                                        x_axis='time', y_axis='mel', fmax=8000,
                                        ax=ax[0])
                    
            fig.colorbar(img, ax=[ax[0]])
           
            ax[0].set(title='Mel Power spectrogram')
            ax[0].label_outer()
            img = librosa.display.specshow(mfccs_, x_axis='time', ax=ax[1])
            fig.colorbar(img, ax=[ax[1]])
            ax[1].set(title='MFCC')
            '''
            librosa.display.waveshow(audio_, sr=sr_, ax=ax[i, 0])  # put wave in row i, column 0
            librosa.display.specshow(mfccs_, x_axis='time', ax=ax[i, 1]) # mfcc in row i, column 1
            librosa.display.specshow(log_s_, x_axis='time', y_axis='log', ax=ax[i, 2])  # spectrogram in row i, column 2
        ipd.display(*audio_players)
