import sys
sys.path.insert(0, '../../')
import matplotlib.pyplot as plt
import tensorflow.keras.saving as keras_saving
from configuration import SAVED_MODEL_PATH
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

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
        self.evaluate()
        return
    def load_model(self):
        return keras_saving.load_model(SAVED_MODEL_PATH+self.file_name)
    
    def evaluate(self):
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

    def get_classification_report(self):
        self.classification_report = classification_report(self.get_flattened_y_test(), 
                                                          self.y_prediction,
                                                          target_names=self.label_encoder.classes_,
                                                          output_dict = True
                                                          )
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
    
 
