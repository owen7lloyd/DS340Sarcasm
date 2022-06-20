from distutils.log import Log
from telnetlib import GA
import numpy as np
from settings import Settings
from data_load import DataLoad, Helper
import os
import json
from collections import Counter

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression



os.chdir('..')
RESULT_FILE = "output/{}.json"
settings = Settings()

def train(train_input, train_output, model):
    clf = make_pipeline(
        StandardScaler() if settings.scale else FunctionTransformer(lambda x: x, validate=False),
        model
    )
    return clf.fit(train_input, np.argmax(train_output, axis=1))

def test(clf, test_input, test_output):
    probas = clf.predict(test_input)
    y_pred = probas
    y_true = np.argmax(test_output, axis=1)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # errors = []

    # for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
    #     if pred != true:
    #         errors.append(idx)

    result_string = classification_report(y_true, y_pred, digits=3)
    print(confusion_matrix(y_true, y_pred))
    print(result_string)
    return classification_report(y_true, y_pred, output_dict=True, digits=3), result_string

def data_setup(data, train_index, test_index):
    train_input, train_output = data.get_split(train_index)
    test_input, test_output = data.get_split(test_index)

    datahelper = Helper(train_input, train_output, test_input, test_output, data)

    train_input = np.empty((len(train_input), 0))
    test_input = np.empty((len(test_input), 0))

    if settings.use_target_text:
        train_input = np.concatenate([train_input, datahelper.get_target_bert_feature(mode="train")], axis=1)
        test_input = np.concatenate([test_input, datahelper.get_target_bert_feature(mode="test")], axis=1)

    if settings.use_target_video:
        train_input = np.concatenate([train_input, datahelper.get_target_video_pool(mode="train")], axis=1)
        test_input = np.concatenate([test_input, datahelper.get_target_video_pool(mode="test")], axis=1)

    if settings.use_target_audio:
        train_input = np.concatenate([train_input, datahelper.get_target_audio_pool(mode="train")], axis=1)
        test_input = np.concatenate([test_input, datahelper.get_target_audio_pool(mode="test")], axis=1)

    if train_input.shape[1] == 0:
        raise ValueError("Invalid modalities")

    if settings.use_author:
        train_input_author = datahelper.get_author(mode="train")
        test_input_author = datahelper.get_author(mode="test")

        train_input = np.concatenate([train_input, train_input_author], axis=1)
        test_input = np.concatenate([test_input, test_input_author], axis=1)

    if settings.use_context:
        train_input_context = datahelper.get_context_bert_features(mode="train")
        test_input_context = datahelper.get_context_bert_features(mode="test")

        train_input = np.concatenate([train_input, train_input_context], axis=1)
        test_input = np.concatenate([test_input, test_input_context], axis=1)

    train_output = datahelper.one_hot_output(mode="train", size=settings.num_classes)
    test_output = datahelper.one_hot_output(mode="test", size=settings.num_classes)

    return train_input, train_output, test_input, test_output

def train_speaker_dependent(model):
    data = DataLoad()
    results = []
    
    for fold, (train_index, test_index) in enumerate(data.get_stratified_k_fold()):
        settings.fold = fold + 1
        print("Present Fold:", settings.fold)
        train_input, train_output, test_input, test_output = data_setup(data, train_index, test_index)
        clf = train(train_input, train_output, model)
        result_dict, result_str = test(clf, test_input, test_output)
        results.append(result_dict)

    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))

    with open(RESULT_FILE.format(settings.model), "w") as file:
        json.dump(results, file)

    print_result(settings.model)

def train_speaker_independent(model):
    data = DataLoad()
    train_index, test_index = data.get_speaker_independent()
    train_input, train_output, test_input, test_output = data_setup(data, train_index, test_index)
    clf = train(train_input, train_output, model)
    test(clf, test_input, test_output)

def speaker_dependent_results(model_num):
    models = {1: LogisticRegression(), 2: SVC(), 3: KNeighborsClassifier(), 4: GaussianProcessClassifier(), 
    5: RandomForestClassifier(), 6: MLPClassifier(), 7: XGBClassifier(), 8: GaussianNB(), 9: QuadraticDiscriminantAnalysis()}

    train_speaker_dependent(models[model_num])

def independent_results():
    model = XGBClassifier()
    train_speaker_independent(model)

def hyperparameter_tune():
    params = []
    results = dict()
    for max_depth in settings.max_depth:
        for learning_rate in settings.learning_rate:
            for gamma in settings.gamma:
                params.append((max_depth, learning_rate, gamma))
    
    for param in params:
        model = XGBClassifier(max_depth=param[0], learning_rate=param[1], gamma=param[2])
        param_str = f"{param[0]} {param[1]} {param[2]}"
        train_speaker_dependent(model)
        results[param_str] = model_result(settings.model)

    print(results)
    return(results)
        

def model_result(model_name):
    with open(RESULT_FILE.format(model_name)) as file:
        results = json.load(file)

    weighted_precision = []
    weighted_recall = []
    weighted_f_scores = []

    for fold, result in enumerate(results):
        weighted_f_scores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])

    return np.mean(weighted_precision), np.mean(weighted_recall), np.mean(weighted_f_scores)

def print_result(model_name):
    with open(RESULT_FILE.format(model_name)) as file:
        results = json.load(file)

    weighted_precision = []
    weighted_recall = []
    weighted_f_scores = []

    print("#" * 20)
    for fold, result in enumerate(results):
        weighted_f_scores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])

        print(f"Fold {fold + 1}:")
        print(f"Weighted Precision: {result['weighted avg']['precision']}  "
              f"Weighted Recall: {result['weighted avg']['recall']}  "
              f"Weighted F score: {result['weighted avg']['f1-score']}")
    print("#" * 20)
    print("Avg :")
    print(f"Weighted Precision: {np.mean(weighted_precision):.3f}  "
          f"Weighted Recall: {np.mean(weighted_recall):.3f}  "
          f"Weighted F score: {np.mean(weighted_f_scores):.3f}")


if __name__ == "__main__":
    model = XGBClassifier(max_depth=3, learning_rate=0.3, gamma=0)
    print(train_speaker_dependent(model))