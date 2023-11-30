import pandas
pandas.options.mode.chained_assignment = None  # default='warn'
import glob
import numpy as np


np.set_printoptions(threshold=np.inf)
import random
import os
from collections import Counter



def main():
    lst1 = glob.glob("/home/xavier/tempDataset/*")
    KDD_collumns = ['age', 'class of worker', 'industry code', 'occupation code',
                                        'education', 'wage per hour', 'enrolled in edu inst last wk', 'marital status', 'major industry code',
                                        'major occupation code', 'race', 'hispanic Origin', 'sex', 'member of a labor union',
                                        'reason for unemployment', 'full or part time employment stat', 'capital gains', 'capital losses',
                                        'divdends from stocks', 'federal income tax liability', 'region of previous residence',
                                        'state of previous residence', 'detailed household and family stat', 'detailed household summary in household','instance weight',
                                        'migration code-change in msa','migration code-change in reg','migration code-move within reg','live in this house 1 year ago',
                                        'migration prev res in sunbelt','num persons worked for employer','family members under 18','country of birth father',
                                        'country of birth mother','country of birth self','citizenship','own business or self employed',
                                        "fill inc questionnaire for veteran's admin",'veterans benefits','weeks worked in year','year','income']

    df = pandas.read_csv("/home/xavier/tempDataset/census-income.data", names=KDD_collumns);
    df2 = pandas.read_csv("/home/xavier/tempDataset/census-income.test", names=KDD_collumns)
    frames=[df,df2]
    finalDf = pandas.concat(frames)

    i=1+2
    finalDf.to_csv("/home/xavier/tempDataset/kdd-income.csv", index=False)
    '''
    lst=sorted(lst)
    categories = ['accuracy', 'TNR', 'TPR', 'BCR', 'CV', 'DIbinary', 'DIavgall', '0-accuracy', '1-accuracy']
    finalDf = pandas.DataFrame(None, columns=['accuracyOriginal', 'accuracyNew', 'accuracyDiference', 'TNROriginal', 'TNRNew', 'TNRDiference', 'TPROriginal', 'TPRNew', 'TPRDiference', 'BCROriginal', 'BCRNew', 'BCRDiference', 'CVOriginal', 'CVNew', 'CVDiference', 'DIbinaryOriginal', 'DIbinaryNew', 'DIbinaryDiference', 'DIavgallOriginal', 'DIavgallNew', 'DIavgallDiference','0-accuracyOriginal','0-accuracyNew','0-accuracyDiference','1-accuracyOriginal','1-accuracyNew','1-accuracyDiference'])
    print(lst)
    for i in range(len(lst)):
        csvName = remove_prefix(lst[i], '/home/xavier/Desktop/Investigação-Fairness/FairnessADASYN/results-100/')
        if os.path.isfile("/home/xavier/.fairness/results/" + csvName):

            #print(csvName)
            try:
                df = pandas.read_csv('/home/xavier/.fairness/results/'+csvName)
                df2 = pandas.read_csv('/home/xavier/Desktop/Investigação-Fairness/FairnessADASYN/results-100/'+csvName)
                #print(df)

                size = len(df)
                if len(df2) < size:
                    size = len(df2)
                finalList = [None]*27
                count = 0
                if size>0:
                    for i in categories:
                        if i in df.columns and i in df2.columns:
                            lst1 = df[i].values.tolist()
                            absLst1 = [abs(ele) for ele in lst1]
                            finalList[count]=sum(absLst1)/len(lst1)

                            lst2 = df2[i].values.tolist()
                            absLst2 = [abs(ele) for ele in lst2]
                            finalList[count+1]=sum(absLst2)/len(lst2)

                            lst3 = [None]*(size)
                            #absLst3=[abs(ele) for ele in lst3]
                            finalList[count+2]=sum(absLst2)/len(lst2)-sum(absLst1)/len(lst1)
                        count=count+3
                    tempDF = pandas.Series(finalList,index=finalDf.columns)
                    tempDF.name=rchop(csvName,".csv")
                    finalDf = finalDf.append(tempDF)
                    finalDf = finalDf.append(pandas.Series(name=''))
                else:
                    print("No rows "+csvName)
            except pandas.errors.EmptyDataError:
                print("OOOPPS")
    for i in range(len(categories)-1):
        finalDf.insert(loc=4*i+3, column="This collum is filling"+str(i), value=['' for j in range(finalDf.shape[0])])

    print("_______________________________________________________")
    print(finalList)
    print("_______________________________________________________")
    print(finalDf)
    finalDf.to_csv('/home/xavier/Desktop/ADASYNDifference.csv')
    finalDf.to_csv('/home/xavier/Desktop/Investigação-Fairness/FairnessADASYN/results-100/ADASYNDifference.csv')
    '''



main()
