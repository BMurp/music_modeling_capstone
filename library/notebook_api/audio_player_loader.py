import sys
sys.path.insert(0, '../../')
import pandas as pd
from library.source_data.feature_extractor import AudioFeatureExtractor
from library.notebook_api.data_loader import  CombinedDataLoader
from configuration import  PROJECT_ABSOLUTE_PATH,MODEL_INPUT_DATA_PATH

from library.notebook_api.audio_player_functions import *

class AudioPlayerLoader():
    def __init__(self, fma_audio_data_size = 'large', tracks_per_genre=4):
        self.in_scope_labels = ['rock', 'electronic', 'hiphop', 'classical', 'jazz','country']
        self.data_loader = CombinedDataLoader(fma_audio_data_size, self.in_scope_labels)
        self.tracks_per_genre = tracks_per_genre
        self.cluster_mapping = {0:"Energetic/Uplifting", 1: 'Warm/Inspiring', 2:'Reflective/Introspective'}
        #default for vector rendering for display
        self.SAMPLE_RATE = 22500
        self.SECONDS = 6
        self.run_audio_extraction()

        return
    def get_modeling_output_df(self):
        clusters = pd.read_csv(MODEL_INPUT_DATA_PATH+'clusters.csv')[['audio_path' , 'label', '0']]
        predicted_genres = pd.read_csv(MODEL_INPUT_DATA_PATH+'predicted_prob.csv')
        combined_df = predicted_genres.merge(clusters, on='audio_path')
        combined_df.drop(columns=['index','label'], inplace=True)
        return combined_df
    
    def get_combined_source_and_model_output_df(self):
        source_data_and_model_output_df = pd.merge(self.data_loader.df_filtered, 
                                                   self.get_modeling_output_df(), 
                                                   on='audio_path')
        df_filtered_in_scope_sample = self.data_loader.get_data_sampled_by_label(self.tracks_per_genre,source_data_and_model_output_df)
        return df_filtered_in_scope_sample

    def run_audio_extraction(self):
        self.audio = AudioFeatureExtractor(self.get_combined_source_and_model_output_df(),
                                           sample_rate=self.SAMPLE_RATE,
                                           start_sample=0,
                                           end_sample=self.SAMPLE_RATE*self.SECONDS)
        self.audio.add_audio_data_to_df()
        self.audio.add_numerical_features_to_df()
        self.audio.add_mfcc_to_df()
        self.audio.add_log_melspectrogram_to_df()
        return
    def get_audio_browser_table_data(self):
        audio_browser_table_data = {
            'Track Id': list(self.audio.df.track_id),  
            'Data Source': list(self.audio.df.dataset),  
            'Audio Player': get_audio_player_html_array(self.audio.df.audio_path.apply(lambda audio_path: PROJECT_ABSOLUTE_PATH+ audio_path)),
            'Log Melspectrogram': get_log_mel_spectrogram_html_image_array(list(self.audio.df.log_melspectrogram)),
            'Feature: Tempo': (list(self.audio.df.features.apply(lambda feature: str(int(feature[10]))))),
            'Actual Genre': list(self.audio.df.label),
            'Explainable Predicted Genre': list(self.audio.df.y_pred),
            'Predicted Probs': get_predicted_probs_stacked_bar_html_image_array(self.audio.df[self.in_scope_labels]),
            'Unsupervised Cluster': list(self.audio.df['0'].apply(lambda x: self.cluster_mapping[x]))

        }
        return audio_browser_table_data
    def get_audio_browser_html(self):
        return get_audio_browser_html(self.get_audio_browser_table_data())