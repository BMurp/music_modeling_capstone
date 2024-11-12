from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix#, accuracy_score, recall_score, 
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
alt.data_transformers.enable("vegafusion")

import sys
sys.path.insert(0, '../../')
from configuration import  MODEL_INPUT_DATA_PATH
from library.notebook_api.data_loader import ModelDataLoader


class ModelScenario():
    def __init__(self,model_data_loader = ModelDataLoader('003'), model = RandomForestClassifier(random_state=42),in_scope_labels = None, in_scope_features = None):
        self.model_data = model_data_loader
        if in_scope_labels == None:
            self.label_names = self.model_data.label_names
            self.df = self.model_data.df

        else:
            self.label_names = in_scope_labels
            self.df = self.model_data.df[self.model_data.df.label.isin(self.label_names)]

        
        if in_scope_features == None:
            self.feature_names = self.model_data.feature_names
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None 
        self.model = None
        self.y_pred = None
        self.model = model
        self.confusion_matrix = None
        self.confusion_matrix_normalized = None
        
        return
      
    
    def get_class_distribution(self):
        return pd.DataFrame(self.df['label'].value_counts(normalize=True) * 100).reset_index()
    
    def get_class_counts(self):
        return self.df.groupby('label')['label'].count().sort_values(ascending=False)

    
    def get_label_sample_df(self,df, label, sample_size):
        df_label = df[df.label == label]
        #return df_label.sample(sample_size).index
        if sample_size > len(df_label):
            return df_label
        return df_label.sample(sample_size)
    
    
    def get_model_data_sampled_by_label(self, sample_size):
        label_sample_indexes = []

        for index, label in enumerate(self.label_names):
            label_sample_df = self.get_label_sample_df(self.model_data.df, self.label_names[index], sample_size)
            #print("Generate ", len(label_sample_df), ' length sample')
            label_sample_indexes.append(label_sample_df)

        sampled_df = pd.concat(label_sample_indexes)
        return sampled_df
    
    def get_feature_distribution_by_label(self, feature_name):
        chart = alt.Chart(self.df).mark_boxplot(extent="min-max").encode(
            alt.X("label:N"),
            alt.Y(feature_name).scale(zero=False),
            alt.Color("label:N").legend(None),
            )
        return chart
    
    def initialize_data(self,df=None):
        if df is not None:
            self.df = df 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.feature_names], self.df['label'], test_size=0.2, random_state=42,stratify=self.df['label'])
        return
    
    def initialize_model(self, model=None):
        if model is not None:
            self.model = model 
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_prob = self.model.predict_proba(self.X_test)
        self.confusion_matrix = confusion_matrix(self.y_test, self.y_pred, labels=self.label_names)
        self.confusion_matrix_normalized = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:,np.newaxis]


    def get_classifcation_report(self):
        return classification_report(self.y_test, self.y_pred)
    
    def chart_confusion_matrix(self, normalized = True):
        cm = None
        fmt = None
        if normalized:
            cm = self.confusion_matrix_normalized
            fmt='.2f'
        else:
            cm=self.confusion_matrix
            fmt = 'd'
        
        sns.heatmap(cm, annot=True,fmt=fmt, cmap='YlGnBu', xticklabels=self.label_names, yticklabels=self.label_names)
        plt.ylabel('Prediction',fontsize=12)
        plt.xlabel('Actual',fontsize=12)
        plt.title('Confusion Matrix',fontsize=16)
        plt.show()

        return 