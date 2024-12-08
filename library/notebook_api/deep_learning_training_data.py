import sys
sys.path.insert(0, '../../')
from library.notebook_api.data_loader import ModelDataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Normalization
from tensorflow import one_hot

def get_feature_vector_data(vector_type = 'mfcc',version_string = '006',vector_length = 1000, apply_normalization = False, apply_resampling=False):
    model_data_loader = ModelDataLoader(version_string)
    model_data_loader_df = model_data_loader.df
    features = []
    vector_height = 0
    most_common_size = 0 
    if version_string == '006':
        most_common_size = 1099
    else:
        most_common_size = 2582

    if vector_type == 'mfcc':
        print('Loading mfcc vectors')
        features = model_data_loader.get_mfcc()
        vector_height = 13
    else:
        print('Loading log mel vectors')
        features = model_data_loader.get_log_melspectrogram()
        vector_height = 128

    labels_series = model_data_loader.df['label']
    #array of feature shapes
    feature_shapes = [feature.shape[1] for feature in features]
    print("features shape distribution")
    print(pd.Series(feature_shapes).value_counts().iloc[0:10])
    print("total records: ", len(features))

    #filter based on features
    # currently filtered to just the most common shapes, as we'll need to normalize shapes prior to training 
    MFCC_LENGTH_CUTOFF = vector_length


    #array of indexes matching a predicate 
    in_scope_feature_indexes = np.where(np.array(feature_shapes) == most_common_size)[0]
    print("normalized length: ",len(in_scope_feature_indexes))

    #filter based on labels
    #pick subset of labels that are more intuitively representative of genre 
    #genres like pop which are ambiguous are removed 
    in_scope_labels = ['rock', 'electronic', 'hiphop', 'classical', 'jazz','country']
    #optain the indexes of label series wher the label is in the list 
    in_scope_label_indexes = np.array(
        (labels_series[
                        labels_series
                            .apply(lambda x: True if x in  in_scope_labels else False)
                    ]
                            .index
        )
    )


    #combine filter for the in scope labels with the same for in scope features 
    in_scope_indexes = [index for index in in_scope_label_indexes if index in in_scope_feature_indexes]
    print("Row Count after label based filter: ", len(in_scope_label_indexes))
    print("Final Row count after label and feature filter: ", len(in_scope_indexes))

    #label encoding
    
    in_scope_label_series = labels_series.iloc[in_scope_indexes]
    unique_label_names = in_scope_label_series.unique()
    unique_label_count = len(in_scope_label_series.unique())
  
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(in_scope_label_series.values)
    encoded_labels = one_hot(indices = numerical_labels, depth = len(unique_label_names))
    
    print("Unique label count: ", unique_label_count)
    print("label data count " , len(encoded_labels))
  
    print("Truncate features to consistent length and reshape")
    #from 1d to 3d 
    features_3d = np.stack(features[in_scope_indexes])
    #clear memory
    features=None
    #truncate to provided input size and reshape to add extra channel for CNN
    reshaped_features = features_3d[:,:,:MFCC_LENGTH_CUTOFF].reshape(features_3d.shape[0],MFCC_LENGTH_CUTOFF,vector_height,1)
    features_3d=None
    #clearning memory space, instantiate original feature array to None 
    features = None
    #generate train and test with stratification for equal label distribution across train and test 
    TEST_SIZE = .2
    print("Generate train_test_split for test size ", TEST_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(np.array(reshaped_features), encoded_labels.numpy(), test_size=TEST_SIZE, random_state=42, stratify =encoded_labels.numpy() )

    #clear input feature array from memory
    reshaped_features = None
    if apply_normalization:
        print('applying normalization')
        # Create a normalization layer
        norm_layer = Normalization()
        # Adapt the layer to your data
        norm_layer.adapt(X_train)

        X_train,X_test= norm_layer(X_train),norm_layer(X_test)

    if apply_resampling:
        print('apply random oversampler on train data')
        oversample = RandomOverSampler(sampling_strategy='not majority', random_state = 42)
        _ = oversample.fit_resample(X_train[:,:,0,0], np.argmax(y_train,axis=1))
        X_train, y_train = np.array(X_train)[oversample.sample_indices_], y_train[oversample.sample_indices_]

    return X_train, X_test, y_train, y_test,label_encoder

#sketch of augmentation
def get_augmented_segment_boundaries(train_row, num_segments=6, overlap_factor=.2 ):
    #based on shape of initial row, and provided segment numbers and overlaps produce array of in and out locations
    full_duration = train_row.shape[0]
    
    equivalized_exact_duration_whole = int(full_duration/num_segments)

    remnant_after_rounding = full_duration - (equivalized_exact_duration_whole *num_segments)

    base_duration_array = [equivalized_exact_duration_whole for index in list(range(0,num_segments)) ]

    padded_duration_array = [int(duration * (1 + overlap_factor)) for duration in base_duration_array]
    start_frame = 0
    end_frame = 0
    segment_boundaries = []
    for index, duration in enumerate(base_duration_array):
        segment_boundary = []
        if index != (num_segments -1):
            end_frame = start_frame + padded_duration_array[index]
        else:
            end_frame = full_duration - 1
            start_frame = end_frame - padded_duration_array[index]
        segment_boundary.append(start_frame)
        segment_boundary.append(end_frame)
        segment_boundaries.append(segment_boundary)
        start_frame += base_duration_array[index]

        #print(index, duration)
    return segment_boundaries


def get_augmented_x_y( X,y, num_segments=8, overlap_factor=.3, row_resample_weight_100_perc_thresh = .9, discount_weight_factor=.25 ):
    #for provided X, y and number of segments to create, and overlap ratio of segments:
    #for each row, determine amount of segments to produce for rows who's sample weights are above a thresh then this will be the full num_segments
    #otherwise apply the discount_weight_factor as mechanism to reduce representation of majority class and increase for minority
    segment_boundaries = get_augmented_segment_boundaries(X[0],num_segments, overlap_factor)
    print("creating augmented data with segment boundaries", segment_boundaries)

    #for oversampling minority
    y_flattened = np.argmax(y,axis=1)
    _, counts = np.unique(y_flattened,return_counts=True)
    #instantiate probability of each label
    label_weights = counts / np.sum(counts)
    label_resample_weights = 1 - label_weights
    print("weights to resample labels are", label_resample_weights)

    augmented_X=[]
    augmented_y = []
    for index, train_row in enumerate(X):
        row_resample_weight = label_resample_weights[y_flattened[index]]
        if row_resample_weight >= row_resample_weight_100_perc_thresh:
            row_resample_weight = 1
        else:
            row_resample_weight = row_resample_weight * discount_weight_factor
        for segment_boundary in segment_boundaries:
            if np.random.binomial(1, row_resample_weight) ==1:
                augmented_X.append(train_row[segment_boundary[0]:segment_boundary[1]])
                augmented_y.append(y[index])
    return np.array(augmented_X), np.array(augmented_y)
