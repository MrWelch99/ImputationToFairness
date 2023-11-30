from copy import deepcopy

import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
import glob
import imblearn
import numpy as np
from sklearn.preprocessing import LabelEncoder

np.set_printoptions(threshold=np.inf)
import random
import os
from collections import Counter

from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s






def main():
    lst = glob.glob("/home/xavier/Desktop/Investigação-Fairness/OldStuff-Investigacao/preprocessed/*.csv")
    lst = sorted(lst)
    for i in range(len(lst)):
        csvName = remove_prefix(lst[i], '/home/xavier/Desktop/Investigação-Fairness/OldStuff-Investigacao/preprocessed/')
        try:
            df = pandas.read_csv('/home/xavier/Desktop/Investigação-Fairness/OldStuff-Investigacao/preprocessed/' + csvName)
            # print(df)

            size = len(df)
            finalList = [None] * 4
            count = 0
            isGerman = False
            isBinary = False
            X=[]
            y=[]


            if "sex-age" in df.columns:
                #print(csvName)
                savedCols = list(df.columns)
                savedCols.remove('sex-age')
                sensititive = "sex-age"
                X = df.loc[:, df.columns != 'sex-age']
                y = np.array(df.loc[:, df.columns == 'sex-age'])
            elif "sex-race" in df.columns:
                #print(csvName)
                savedCols = list(df.columns)
                savedCols.remove('sex-race')
                sensititive = "sex-race"
                X = df.loc[:, df.columns != 'sex-race']
                y = np.array(df.loc[:, df.columns == 'sex-race'])
            elif "race-sex" in df.columns:
                #print(csvName)
                savedCols = list(df.columns)
                savedCols.remove('race-sex')
                sensititive = "race-sex"
                X = df.loc[:, df.columns != 'race-sex']
                y = np.array(df.loc[:, df.columns == 'race-sex'])
            elif "sex" in df.columns and "age" in df.columns:
                #print(csvName)
                #print(savedCols)
                savedCols = list(df.columns)
                savedCols.remove("sex")
                savedCols.remove("age")
                sensititive = "sex"
                sensititive2 = "age"
                temp = df.loc[:, (df.columns != 'sex')]
                X = temp.loc[:, temp.columns != 'age']
                y1 = np.array(df.loc[:, df.columns == 'sex'])
                y2 = np.array(df.loc[:, df.columns == 'age'])


                ageTemp = np.copy(y2)

                for k in ageTemp:
                    if k[0] >= 25:
                        k[0] = 0
                    else:
                        k[0] = 1

                y = np.copy(y1)
                y=y.astype('U64')

                for k in range(len(y)):
                    #print("...."+str(y[k][0])+"+"+str(ageTemp[k][0])+"="+(str(y[k][0]) + str(ageTemp[k][0])))
                    y[k][0] = str(y[k][0])+str(ageTemp[k][0])
                    #--------------------------------ACABAR ISTO--------------------------------------------
                isGerman = True
            elif "Race" in df.columns:
                #print(csvName)
                savedCols = list(df.columns)
                savedCols.remove('Race')
                sensititive = 'Race'
                X = df.loc[:, df.columns != 'Race']
                y = np.array(df.loc[:, df.columns == 'Race'])
                #print(savedCols)

            if len(X)!=0 and len(y)!=0:
                dict = {}


                #print(csvName)
                for j in range(len(y)):
                    if y[j][0] in dict.keys():
                        dict[y[j][0]] = dict[y[j][0]] + 1
                    else:
                        dict[y[j][0]] = 1
                    count = count + 1
                #print(dict)
                ratio = count / len(dict) + 0.10 * count
                finalDict = {}
                #print(finalDict)

                maxValue = max(dict.values())

                countfinalDict=0

                
                if maxValue > ratio:
                    for j in dict:
                        #APONTAR DATASET DIFFERENCES
                        if dict[j] < 6:
                            finalDict[j] = dict[j]
                        else:
                            finalDict[j] = maxValue
                        countfinalDict += finalDict[j]

                print("Original Length")
                print(X.shape[0])

                print("Final Dict")
                print(countfinalDict)


                if bool(finalDict):
                    print(csvName)
                    categoricalDict = np.full(X.shape[1], False)

                    hasCategorical=False
                    for k in range(X.shape[1]):
                        if isinstance(X.iloc[0,k],str):
                            categoricalDict[k]=True
                            hasCategorical=True

                    label_encoder_array=[None]*categoricalDict.size;

                    trial=-1;
                    for k in range(categoricalDict.size):
                        if categoricalDict[k]==True:
                            label_encoder_array[k]=LabelEncoder();

                            #temp=deepcopy(X.iloc[:, k]);

                            X.iloc[:, k] = label_encoder_array[k].fit_transform(X.iloc[:, k]);

                            #print(X.iloc[:, k]);
                            trial=k;
                    #if trial!=-1:
                    #    print(temp)
                    #    print(X.iloc[:, trial])

                    ada = ADASYN(sampling_strategy=finalDict);

                    #print(X)
                    print(X.shape)
                    print(y.shape)
                    print(categoricalDict)
                    print(dict)
                    print(finalDict)

                    X_Smote, y_Smote = ada.fit_resample(X, y)



                    for k in range(categoricalDict.size):
                        if categoricalDict[k]==True:
                            X_Smote.iloc[:,k]=label_encoder_array[k].inverse_transform(X_Smote.iloc[:,k]);
                            #print("/////////////////////////////////////////////////////////////////////////////////7");
                            #print(X.iloc[:,k])
                    #if trial!=-1:
                    #    print(X_Smote.iloc[:, trial])

                    # print(X)
                    # print(X_Smote)
                    # print(Counter(y_Smote))
                    if isGerman:
                        #numAdd=final
                        savedCols.append(sensititive)
                        savedCols.append(sensititive2)

                        for j in range(len(y_Smote)):
                            #print("...."+str(j)+"...."+str(len(y_Smote))+"....."+str(len(y1)))
                            if (j+2)>len(y1):
                                #print("YEEEEEEEEEEEEEEEEEEEEEEEY")
                                #print([int(y_Smote[j][0:len(y_Smote[j])-1])])
                                if y1.dtype == 'int64':
                                    y1 = np.append(y1, [[int(y_Smote[j][0:len(y_Smote[j])-1])]], axis=0)
                                else:
                                    y1 = np.append(y1, [[y_Smote[j][0:len(y_Smote[j]) - 1]]], axis=0)
                            if (j+2)>len(y2):
                                #print("YEEEEEEEEEEEEEEEEEEEEEEEY")
                                check=int(y_Smote[j][len(y_Smote[j]) - 1:len(y_Smote[j])])
                                #print(check)
                                if check == 0:
                                    rand = random.randint(26,65)
                                    y2 = np.append(y2, [[rand]], axis=0)
                                else:
                                    rand = random.randint(15, 24)
                                    y2 = np.append(y2, [[rand]], axis=0)
                        #print(y2)
                        finalDf = pandas.concat([pandas.DataFrame(X_Smote), pandas.DataFrame(y1),pandas.DataFrame(y2)], axis=1)
                        #print(finalDf)
                        #-------------------------CONTINUE SEE COLLUMNS
                    else:
                        savedCols.append(sensititive)
                        finalDf = pandas.concat([pandas.DataFrame(X_Smote), pandas.DataFrame(y_Smote)], axis=1)
                    finalDf.columns = savedCols
                    # print("")
                    # print(lst)
                    # print("_______________________________________________________")
                    print(finalDf)
                    finalDf.to_csv('/home/xavier/Desktop/Fairness/fairness/data/preprocessed/' + csvName, index=False)
                    '''except ValueError:
                        #------------------------------------------------------------VER ESTA MERDA! EU SOU RETARDADO------------------------------
                        print("ValueError")'''

        except pandas.errors.EmptyDataError:
            print("OOOPPS")

    # finalDf.to_csv('/home/xavier/Desktop/Fairness/fairness/data/preprocessed/'+csvName+'.csv')


main()
