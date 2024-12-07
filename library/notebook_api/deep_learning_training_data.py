import sys
sys.path.insert(0, '../../')
from library.notebook_api.data_loader import ModelDataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Normalization
from tensorflow import one_hot

def get_feature_vecor_data(vector_type = 'mfcc',version_string = '005',vector_length = 1200, apply_normalization = False):
    model_data_loader = ModelDataLoader(version_string)
    model_data_loader_df = model_data_loader.df
    features = []
    vector_height = 0
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
    #MFCC_LENGTH_CUTOFF = 2582
    #MFCC_LENGTH_CUTOFF = 500
    MFCC_LENGTH_CUTOFF = vector_length


    #array of indexes matching a predicate 
    in_scope_feature_indexes = np.where(np.array(feature_shapes) >= MFCC_LENGTH_CUTOFF)[0]
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
    '''
    
    label_to_int_map = {}
    for index, label in enumerate(unique_label_names):
        label_to_int_map[label] = index  


    int_to_label_map = {v: k for k, v in label_to_int_map.items()}

    #numerical_labels = labels_series.map(label_to_int_map)
    numerical_labels = in_scope_label_series.map(label_to_int_map)
    '''
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(in_scope_label_series.values)


    #encoded_labels = one_hot(indices = numerical_labels.values, depth = len(unique_label_names))

    encoded_labels = one_hot(indices = numerical_labels, depth = len(unique_label_names))
    
    print("Unique label count: ", unique_label_count)
    print("label data count " , len(encoded_labels))

    #truncate features to consistent length while reshaping 
    reshaped_features = []
    feature_array = []
    for feature in features[in_scope_indexes]:
        mfcc = [] 
        for vector in feature: 
            mfcc.append(vector[0:MFCC_LENGTH_CUTOFF])
            mfcc_reshaped = np.array(mfcc)
        feature_array.append(mfcc_reshaped)
        reshaped_features.append(mfcc_reshaped.reshape((MFCC_LENGTH_CUTOFF,vector_height,1)))
    
    #generate train and test with stratification for equal label distribution across train and test 
    TEST_SIZE = .2

    X_train, X_test, y_train, y_test = train_test_split(np.array(reshaped_features), encoded_labels.numpy(), test_size=TEST_SIZE, random_state=42, stratify =encoded_labels.numpy() )
    if apply_normalization:
        print('applying normaization')
        # Create a normalization layer
        norm_layer = Normalization()
        # Adapt the layer to your data
        norm_layer.adapt(X_train)

        X_train,X_test= norm_layer(X_train),norm_layer(X_test)
    return X_train, X_test, y_train, y_test,label_encoder