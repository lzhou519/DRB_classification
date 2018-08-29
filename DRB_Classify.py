import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb

def crit_features(vect, feature_chi2_ind, SVM, X_train_dtm, y_train):
    """Finding the critical features with the classifiers coefficients"""
    SVM.fit(X_train_dtm, y_train)
    clf_coefs = SVM.coef_
    clf_coefs = clf_coefs[0]

    """finding the index of the important coefficients and their value and make a list of it"""
    imp_coeff_index = []
    coeffs = []

    for i in range(len(clf_coefs)):
        if clf_coefs[i] != 0:
            imp_coeff_index.append(i)
            coeffs.append(clf_coefs[i])

    """edit the feature names to mimic the feature reduction after chi2 and SVM"""
    feature_names = vect.get_feature_names()
    feature_names = np.array(feature_names)
    feature_names = feature_names[feature_chi2_ind]
    feature_names = feature_names[imp_coeff_index]

    """Sorting the coeffs and feature names"""
    sorted_ind_coeffs = np.argsort(coeffs)
    sorted_coeffs = np.sort(coeffs)
    sorted_feature_names = feature_names[sorted_ind_coeffs]

    """Puting the sorted coeffs and features in to a dataframe"""
    sorted_coeffs_col = pd.DataFrame({'sorted_coeff': sorted_coeffs})
    sorted_feature_names_col = pd.DataFrame({'sorted_feature_names': sorted_feature_names})

    df = pd.DataFrame()
    df = pd.concat([df, sorted_coeffs_col, sorted_feature_names_col],
                   axis=1)
    df.to_csv('Extracted_Features_chiLinearSVC.csv', encoding='cp1252')
    return feature_names.tolist(), imp_coeff_index, coeffs

def Metrics(y_test, predicted):
    accuracy = metrics.accuracy_score(y_test, predicted)
    precision = metrics.precision_score(y_test, predicted)
    recall = metrics.recall_score(y_test, predicted)
    f1 = metrics.f1_score(y_test, predicted)
    return accuracy, precision, recall, f1

"""starting file, data extracted with pandas."""
Data = pd.read_csv('Data_Filtered_all.csv', encoding='cp1252')

"""separating data for Training and Testing"""
Data['ISSUE_DATE'] = pd.to_datetime(Data['ISSUE_DATE'], errors='coerce')
Training_Data = Data[Data['ISSUE_DATE'] <= '2017-08']
Training_Data = Training_Data.reset_index(drop=True)

Testing_Data = Data[Data['ISSUE_DATE'] > '2017-08']
Testing_Data = Testing_Data.reset_index(drop=True)
Testing_Data.to_csv('Testing_Data.csv', encoding='cp1252')

X_train = Training_Data.New_Combined
y_train = Training_Data.EXPORT_STATUS_num

X_test = Testing_Data.New_Combined
y_test = Testing_Data.EXPORT_STATUS_num
"""finish of separation of data for Training and Testing"""

mindif = 4
maxdif = 0.4
ngram = (1, 1)
NLR_prob = .96

"""Using CountVectorizer to make a document term matrix."""
"""Then use of transform to ready the X train and test matrix for classification."""
vect = CountVectorizer(stop_words='english', min_df=mindif,
                       max_df=maxdif, ngram_range=ngram)

vect.fit(X_train, y_train)
X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

"""feature selection with chi2"""
ch2 = SelectKBest(chi2, k=5500)
X_train_dtm = ch2.fit_transform(X_train_dtm, y_train)
X_test_dtm = ch2.transform(X_test_dtm)
feature_chi2_ind = ch2.get_support(indices=True)

"""Use classifier"""
SVM = LinearSVC(C=.02, penalty="l1", dual=False, class_weight='balanced')

"""Finding the critical features with function crit_features"""
feature_names, imp_coeff_index, coeffs = crit_features(vect, feature_chi2_ind, SVM, X_train_dtm, y_train)

"""Wrap CalibratedClassifierCV around SVM to get predict_proba"""
clf = CalibratedClassifierCV(SVM)
clf.fit(X_train_dtm, y_train)

"""Use the probabilities to make a list of indexes from the test source that can be classified."""
probabilities = clf.predict_proba(X_test_dtm)
prob_list = []

"""Find the index of the rows where the probabilities is higher than .96 that it is NLR"""
for i in range(len(probabilities)):
    if probabilities[i, 1] > NLR_prob:
        prob_list.append(i)

"""Edit the X_test to have only the rows that have a probability greater than .96"""
X_test = X_test[prob_list]

"""Edited X_test_dtm to a dense matrix for later use"""
X_test_dense = X_test_dtm.toarray()
X_test_dense = X_test_dense[:, imp_coeff_index]
X_test_dense = X_test_dense[prob_list, :]

X_test_dtm_filtered = X_test_dtm[prob_list , :]
y_test = y_test[prob_list]

"""calculate y_pred with clf.predict"""
y_pred = clf.predict(X_test_dtm_filtered)

"""use Metrics function to find the evaluation values below"""
accuracy, precision, recall, f1 = Metrics(y_test, y_pred)
metric = metrics.confusion_matrix(y_test, y_pred)

"""Compiling a list of articles that were classified as false positive"""
FP = X_test[(y_pred == 1) & (y_test == 0)]
print(FP)

"""use prob_list to find the index of the test articles."""
"""use FP_index to find the index of the articles classified as FP."""
FP_index = []
for i in range(len(prob_list)):
    if (y_pred[i] == 1) and (y_test[prob_list[i]] == 0):
        FP_index.append(prob_list[i])

"""article_SigInfo shows the signification features for each article/row"""
article_SigInfo = []
for row in range(X_test_dense.shape[0]):
    row_info = []

    X_test_row = X_test_dense[row, : ].tolist()
    for i, val in enumerate(X_test_row):
        if val != 0:

            row_info.append((feature_names[i], val, np.sign(coeffs[i])))

    article_SigInfo.append(row_info)

article_SigInfo = {'Sig_info': article_SigInfo}
article_SigInfo = pd.DataFrame(article_SigInfo)

Testing_Data_Classified = Testing_Data.iloc[prob_list]
Testing_Data_Classified = Testing_Data_Classified.reset_index(drop=True)

Testing_Data_Sig = pd.concat([Testing_Data_Classified, article_SigInfo], axis=1)
Testing_Data_Sig.to_csv('Testing_Data_Sig.csv', encoding='cp1252')

print(FP_index)

print(metric)
print('accuracy', accuracy)
print('precision', precision)
print('recall', recall)
print('f1', f1)
print('percentage classified', len(prob_list)/len(Testing_Data))

print('done')