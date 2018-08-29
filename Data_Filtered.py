import pandas as pd
import re
import unicodedata
import string
import re

def Filtered(Data):

    "Initial Filtering"
    Data = Data.drop('ARTICLE_WF_STATE_SEQ_ID', 1)
    Data = Data.drop('LIBRARY_SEQ_ID', 1)
    Data = Data.drop('ARTICLE_SEQ_ID', 1)
    Data = Data.dropna(subset=['EXPORT_STATUS'])
    Data = Data.reset_index(drop=True)
    Data = Data.dropna(subset=['CLASS_DESC_STRING'])
    Data = Data.reset_index(drop=True)
    Data = Data.where((pd.notnull(Data)), ' ')
    Data = Data.drop(Data[Data.EXPORT_STATUS == 'UNK'].index)
    Data = Data.reset_index(drop=True)
    Data = Data.drop(Data[Data.EXPORT_STATUS == 'NLR+LRJ'].index)
    Data = Data.reset_index(drop=True)

    "Make new column the gives a numeric value for export control classifications"
    Data['EXPORT_STATUS_num'] = Data.EXPORT_STATUS.map({'NLR': 1, 'LRC': 0, 'LRS': 0, \
                                                        'NLR+LRJ': 0, \
                                                        'LRS+LRC': 0, 'LRC+LRJ': 0, \
                                                        'NSR': 1, 'LR': 0, 'DNE': 0,\
                                                        'NLR~NLR':1, 'LRS~LRS':0})

    """
    Data['EXPORT_STATUS_num'] = Data.EXPORT_STATUS.map({'NLR': 0, 'LRC': 1, 'LRS': 2, \
                                                        'NLR+LRJ': 4, \
                                                        'LRS+LRC': 4, 'LRC+LRJ': 1, \
                                                        'NSR': 0, 'LR': 4, 'DNE': 4,\
                                                        'NLR~NLR':0, 'LRS~LRS':2})
                                                        """


    "Filtering on 9E003.a"
    #Data['ExportSpec'] = Data["CLASS_DESC_STRING"].str.contains("9e003.a", case=False)
    #Data['ExportSpec_TF'] = Data.ExportSpec.map({True: 1, False: 0})

    "Row filtering focusing on review area"
    #Data = Data.drop(Data[(Data.REVIEW_AREA_SEQ_ID != 40) \
    #                      & (Data.REVIEW_AREA_SEQ_ID != 31) \
    #                      & (Data.REVIEW_AREA_SEQ_ID != 27)].index)

    "Dropping more columns"
    Data = Data.drop('LEGACY_ARTICLE_CODE', 1)
    Data = Data.drop('OBJECT_DESCRIPTION', 1)
    Data = Data.drop('REASON_FOR_CLASSIFICATION', 1)
    Data = Data.drop('APP_OBJECT_ID', 1)
    Data = Data.drop('LEGACY_IND', 1)
    Data = Data.drop('REVIEW_AREA_SEQ_ID', 1)
    Data = Data.drop('ENGINE_MODEL_SEQ_ID', 1)
    Data = Data.drop('DISCIPLINE_AREA_SEQ_ID', 1)

    Data = Data.drop('OWNER_USER_ID', 1)
    Data = Data.drop('REVIEW_DATE', 1)
    Data = Data.drop('LAST_UPDATED_BY', 1)
    Data = Data.drop('LAST_UPDATE_DATE', 1)
    Data = Data.drop('PART_NUMBER', 1)
    Data = Data.drop('ARTICLE_ID', 1)
    Data = Data.drop('OWNING_BUSINESS_SEGMENT_SEQ_ID', 1)

    "Filtering out words like none or see attachments"
    Data = Data.replace({'See Attachments': ' '}, regex=True)
    Data = Data.replace({'See Attachment': ' '}, regex=True)
    Data = Data.replace({'See attached': ' '}, regex=True)
    Data = Data.replace({'Not applicable': ' '}, regex=True)
    Data = Data.replace({'None': ' '}, regex=True)
    Data = Data.replace({'none': ' '}, regex=True)

    "Combining Columns"
    Data["Combined"] = Data["ARTICLE_TITLE"] + ' ' + \
                       Data["KEY_WORDS"] + ' ' + \
                       Data["ENGINE_MODEL_NAME"] + ' ' + \
                       Data["DISCIPLINE_AREA_NAME"] + ' ' + \
                       Data["REVIEW_AREA_NAME"] + ' ' + \
                       Data["ARTICLE_SUBJECT"] + ' ' + \
                       Data["ARTICLE_ABSTRACT"] + ' ' + \
                       Data["PROBLEM_PURPOSE"] + ' ' + \
                       Data["REFERENCES"] + ' ' + \
                       Data["ASSUMPTIONS"] + ' ' + \
                       Data["RESULTS"] + ' ' + \
                       Data["CONCLUSIONS"]

    "DRB Header Info"
    #Data = Data.dropna(subset=['ENGINE_MODEL_NAME'])
    #Data = Data.dropna(subset=['DISCIPLINE_AREA_NAME'])
    #Data = Data.dropna(subset=['REVIEW_AREA_NAME'])

    #Data['ENGINE_MODEL_SEQ_ID'] = Data.ENGINE_MODEL_SEQ_ID.astype(int)
    #Data['ENGINE_MODEL_SEQ_ID'] = Data.ENGINE_MODEL_SEQ_ID.astype(str)
    #Data["Headers_Combined"] = Data["ENGINE_MODEL_SEQ_ID"] + ' ' + \
    #                   Data["DISCIPLINE_AREA_NAME"] + ' ' + \
    #                   Data["REVIEW_AREA_NAME"]


    "Dropping the combined columns"
    Data = Data.drop('ARTICLE_TITLE', 1)
    Data = Data.drop('ARTICLE_SUBJECT', 1)
    Data = Data.drop('ARTICLE_ABSTRACT', 1)
    Data = Data.drop('PROBLEM_PURPOSE', 1)
    Data = Data.drop('REFERENCES', 1)
    Data = Data.drop('ASSUMPTIONS', 1)
    Data = Data.drop('RESULTS', 1)
    Data = Data.drop('CONCLUSIONS', 1)
    Data = Data.drop('ENGINE_MODEL_NAME', 1)
    Data = Data.drop('KEY_WORDS', 1)

    "Reset Index"
    Data = Data.reset_index(drop=True)

    return Data

def Depunc(row, dictionary, dict_regex, html_xml_regex):
    CombinedText = row["Combined"].lower()

    normalized = unicodedata.normalize('NFKD', CombinedText)

    """Transform accentuated unicode symbols into their simple counterpart
    Warning: the python-level loop and join operations make this
    implementation 20 times slower than the strip_accents_ascii basic
    normalization.
    See also
    --------
    strip_accents_ascii
        Remove accentuated char for any unicode symbol that has a direct
        ASCII equivalent."""
    if normalized == CombinedText:
        CombinedText = normalized
    else:
        CombinedText = ''.join([c for c in normalized if not unicodedata.combining(c)])

    """Transform accentuated unicode symbols into ascii or nothing
        Warning: this solution is only suited for languages that have a direct
        transliteration to ASCII symbols."""
    CombinedText = unicodedata.normalize('NFKD', CombinedText).encode('ASCII', 'ignore').decode('ASCII')

    """Basic regexp based HTML / XML tag stripper function"""
    CombinedText = html_xml_regex.sub(" ", CombinedText)

    """Replace abbreviations of words"""
    CombinedText = dict_regex.sub(lambda mo: dictionary[mo.string[mo.start():mo.end()]], CombinedText)

    for char in string.punctuation:
        CombinedText = CombinedText.replace(char, ' ')

    # split the string on the spaces. Get a resulting list with strings
    TokenList = CombinedText.split()

    TokenListResult = []

    for Token in TokenList:
        if Token.isdigit():
            TokenListResult.append('')
        else:
            TokenListResult.append(Token)

    #row['Token_Lengths'] = len(TokenListResult)
    """combine the list of strings in to a string"""
    row['New_Combined'] = " ".join(TokenListResult)

    return row

# Using pandas to read the .csv data files DRB_example.csv is an example of what would be in DRBs_all.csv
# Equiv_Words.csv is included
# DRB_Tags.csv is not included for export control reasons.
Tag = pd.read_csv('DRB_Tags.csv', encoding='cp1252', low_memory=False)
Data = pd.read_csv('DRBs_all.csv', encoding='cp1252', low_memory=False)
equiv_wd = pd.read_csv('Equiv_Words.csv', encoding='UTF-8')

"""Merge Data with Classification"""
Data = Data.merge(Tag, left_on='ARTICLE_CODE', right_on='APP_OBJECT_ID', how='left')

"""Making the equivalent words in to a dictionary"""
dictionary = dict(zip(equiv_wd.equiv_word, equiv_wd.target_word))

# using the Filtered function to do some basic filtering of the Data.
Data = Filtered(Data)

# using regex to filter out and replace unicodes, and abbreviations from the Equiv_Words.csv
# regex is first comipled with re.compile then used in Depunc function
dict_regex = re.compile("(%s)" % "|".join(map(lambda x: r'\b' + x + r'\b', dictionary.keys())))
html_xml_regex = re.compile(r"<([^>]+)>", flags=re.UNICODE)
Data = Data.apply(Depunc, dictionary = dictionary, dict_regex = dict_regex, html_xml_regex= html_xml_regex, axis=1)

# some more filtering
Data = Data.drop('Combined', 1)
Data = Data.drop(Data[Data.New_Combined.str.contains('to be added by custodian none')].index)
Data = Data.drop(Data[Data.New_Combined.str.contains('no design summary none')].index)
Data = Data.drop(Data[Data.New_Combined.str.contains('digitized paper study')].index)
Data = Data.reset_index(drop=True)

print(Data.shape)
print(Data.EXPORT_STATUS.value_counts())
print(Data.EXPORT_STATUS_num.value_counts())

# ending file to be used for classification, no file will be in Github due to export control reasons.
Data.to_csv('Data_Filtered_all.csv', encoding='cp1252', index=False)

print('done')


