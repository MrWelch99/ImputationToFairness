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

    file1 = open("/home/xavier/tempDataset/communities.names", 'r')
    Lines = file1.readlines()
    Lines = [line.rstrip() for line in Lines]
    df = pandas.read_csv("/home/xavier/tempDataset/communities.data",names=Lines);
    #df = pandas.read_csv("/home/xavier/tempDataset/default of credit card clients.xls")
    df.insert(0, "race","")


    #df["race"] = ""
    black = df['racepctblack'] >= 0.06
    df.loc[black, "race"] = 'black'
    not_black = df['race'] != 'black'
    df.loc[not_black, 'race'] = 'not_black'
    print("Black :"+str(sum(black)))
    print("Not Black :" + str(sum(not_black)))

    high_crime = df['ViolentCrimesPerPop'] >= 0.7
    df.loc[high_crime, "ViolentCrimesPerPop"] = 'high_crime'
    low_crime = df['ViolentCrimesPerPop'] != 'high_crime'
    df.loc[low_crime, 'ViolentCrimesPerPop'] = 'low_crime'
    print("High Crime :" + str(sum(high_crime)))
    print("Low Crime :" + str(sum(low_crime)))


    print(df)
    print(df.columns)
    i=1+1
    df.to_csv("/home/xavier/tempDataset/communities.csv", index=False)
    #df2 = pandas.read_csv("/home/xavier/tempDataset/census-income.test", names=KDD_collumns)
    #frames=[df,df2]
    #finalDf = pandas.concat(frames)

    #i=1+2
    #df.to_csv("/home/xavier/tempDataset/kdd-income.csv", index=False)

main()