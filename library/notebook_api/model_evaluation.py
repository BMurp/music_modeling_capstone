import sys
sys.path.insert(0, '../../')
import matplotlib.pyplot as plt
import tensorflow.keras.saving as keras_saving
from configuration import SAVED_MODEL_PATH
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import altair as alt
alt.data_transformers.enable("vegafusion")
alt.renderers.enable('default')


def plot_training_history(history):
    fig, axs = plt.subplots(2)
    fig.set_size_inches(12, 8)
    fig.suptitle('Training History', fontsize=16)
    axs[0].plot(history.epoch, history.history['loss'], history.history['val_loss'])
    axs[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axs[0].legend(['loss', 'val_loss'])
    axs[1].plot(history.epoch, history.history['accuracy'], history.history['val_accuracy'])
    axs[1].set(title='Accuracy', xlabel='Epoch', ylabel='Accuracy')
    axs[1].legend(['accuracy', 'val_accuracy'])
    plt.show()


class ModelEvaluation():
    def __init__(self,name, file_name, X_test, y_test, label_encoder):
        self.name = name
        self.file_name = file_name
        self.model = self.load_model()
        self.X_test = X_test
        self.y_test = y_test 
        self.label_encoder = label_encoder      
        self.y_prediction = np.argmax(self.model.predict(x=X_test), axis=1)
        #self.evaluate()
        return
    def load_model(self):
        return keras_saving.load_model(SAVED_MODEL_PATH+self.file_name)
    
    def evaluate(self):
        print("Evaluating Model: ", self.name)
        self.display_accuracy_and_loss()
        _ = self.get_classification_report()
        self.display_confusion_matrix()

    def display_accuracy_and_loss(self):
        # Evaluate the model on the test set
        self.test_loss, self.test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print(f"\nTest Accuracy: {self.test_accuracy * 100:.2f}%")
        print(f"Test Loss: {self.test_loss:.4f}")
        return
    
    def get_flattened_y_test(self):
        #flattens , if needed y_test to account for differnet shapes across scenarios
        y_test_ = None
        if len(self.y_test.shape) > 1:
            y_test_ = np.argmax(self.y_test,axis=1)
        else:
            y_test_ = self.y_test 
        
        return y_test_

    def get_classification_report(self, print_report = True):
        self.classification_report = classification_report(self.get_flattened_y_test(), 
                                                          self.y_prediction,
                                                          target_names=self.label_encoder.classes_,
                                                          output_dict = True
                                                          )
        if print_report:
            print("\nClassification Report:")
            print(classification_report(self.get_flattened_y_test(), 
                                                            self.y_prediction,
                                                            target_names=self.label_encoder.classes_
                                            
                                                            ))
        return self.classification_report
    
    def display_confusion_matrix(self):
        # Confusion matrix
        cm = confusion_matrix(self.get_flattened_y_test(), self.y_prediction)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_, cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        return
    

class ModelEvaluationComparisons():
    #for comparing multiple models
    def __init__(self,model_evals):
        self.model_evals = model_evals
        self.metrics = ['precision', 'recall','f1-score']
        self.genres = model_evals[0].label_encoder.classes_
        self.comparison_dfs = self.get_model_evaluation_comparison_data()
        return
    def get_model_evaluation_comparison_data(self):
        comparison_dfs = {}
        for metric in self.metrics:
            metric_model_array = []
            for model in self.model_evals:
                metric_model_array.append(pd.DataFrame(model.get_classification_report(print_report=False)).T[metric])

            comparison_dfs[metric] = pd.concat(metric_model_array, axis=1)
            comparison_dfs[metric].columns = [model.name for model in self.model_evals]
            comparison_dfs[metric] = comparison_dfs[metric].T
            comparison_dfs[metric].reset_index(inplace=True)
        return comparison_dfs
    def display_metric_comparison_chart(self,metric_name):
        return alt.Chart(self.comparison_dfs[metric_name]).mark_bar().encode(
                        x=alt.X('amount:Q', title=None),
                        y=alt.Y('type:N', title=None),
                        color=alt.Color('amount:Q', legend=None),
                        column=alt.Column('index',
                                        title=None)
                        ).transform_fold(as_=['type', 'amount'],
                                            fold=self.genres).properties(
                                                width=100,
                                                title=f'{metric_name} by Model')
    
    def display_metric_comparison_charts(self):
        for metric in self.metrics:
            self.display_metric_comparison_chart(metric)

 
