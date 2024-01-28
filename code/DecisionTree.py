from sklearn import tree
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import pandas as pd

class DecisionTree:

    def __init__(self,criterion,split_criterion,depths_to_test,fold_number,X_train,X_test,y_train,y_test):

        self.criterion = criterion
        self.splitter = split_criterion
        self.depths_to_test=depths_to_test
        self.fold_number=fold_number
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        #------cv metrics-----#

        self.cv_accuracies = {}
        self.cv_precisions = {}
        self.cv_recalls = {}
        self.cv_f1_scores = {}
        self.cv_conf_matrices = {}

    def create_model(self,max_depth):

        return tree.DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=max_depth
        )


    def execute_k_fold_cross_validation(self):

        for depth in self.depths_to_test:

            dt = self.create_model(max_depth=depth)

            y_pred = cross_val_predict(dt,self.X_train,self.y_train,cv=self.fold_number)

            accuracy = accuracy_score(self.y_train,y_pred)
            self.set_cv_accuracies(depth,accuracy)

            precision = precision_score(self.y_train, y_pred)
            self.set_cv_precisions(depth,precision)
            
            recall = recall_score(self.y_train, y_pred)
            self.set_cv_recalls(depth,recall)
            
            f1 = f1_score(self.y_train, y_pred)
            self.set_cv_f1_scores(depth,f1)
            
            conf_matrix = confusion_matrix(self.y_train, y_pred)
            self.set_cv_conf_matrices(depth,conf_matrix)

    #---------------------GETTER--------------------------------#

    def get_cv_metrics(self):

        return pd.DataFrame({
            'Profondità': self.depths_to_test,
            'Accuratezza': [self.cv_accuracies[depth] for depth in self.depths_to_test],
            'Precisione': [self.cv_precisions[depth] for depth in self.depths_to_test],
            'Recall': [self.cv_recalls[depth] for depth in self.depths_to_test],
            'F1-Score': [self.cv_f1_scores[depth] for depth in self.depths_to_test],
            'Confusion Matrix': [self.cv_conf_matrices[depth] for depth in self.depths_to_test],
        })
    
   
    
    #------------------------SETTER-------------------------------#
    
    def set_cv_accuracies(self,depth,accuracy):

        if depth is not None and accuracy is not None:

            self.cv_accuracies[depth] = accuracy

    def set_cv_precisions(self,depth,precision):

        if depth is not None and precision is not None:

            self.cv_precisions[depth] = precision

    def set_cv_recalls(self,depth,recall):

        if depth is not None and recall is not None:

            self.cv_recalls[depth] = recall

    def set_cv_f1_scores(self,depth,f1_score):

        if depth is not None and f1_score is not None:

            self.cv_f1_scores[depth] = f1_score

    def set_cv_conf_matrices(self,depth,conf_matrix):

        if depth is not None and conf_matrix is not None:

            self.cv_conf_matrices[depth] = conf_matrix



            

    


    

    