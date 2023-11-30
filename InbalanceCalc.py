import pandas
import glob
import os


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def rchop(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s

def main():
    lst = glob.glob("/home/xavier/Desktop/Fairness/fairness/data/preprocessed/*.csv")
    lst=sorted(lst)
    finalDf = pandas.DataFrame(None,columns=['Sensitive-atr', 'Type', 'Occurrences', 'Ratio'])
    for i in range(len(lst)):
        str = remove_prefix(lst[i], '/home/xavier/Desktop/Fairness/fairness/data/preprocessed/')
        try:
            df = pandas.read_csv('/home/xavier/Desktop/Fairness/fairness/data/preprocessed/' + str)

            #print(df)
            print("\n")

            size = len(df)
            finalList = [None] * 4
            count = 0

            if "sex-age" in df.columns:
                tempdf=df.pivot_table(index=['sex-age'], aggfunc='size')
                #print(len(df))
                #print()
                #print(tempdf)
                #print(tempdf[0])
                for j in range(tempdf.size):
                    finalList[0]="sex-age"
                    finalList[1]=tempdf.index[j]
                    finalList[2]=tempdf[j]
                    finalList[3]=tempdf[j]/len(df)
                    tempDF = pandas.Series(finalList, index=finalDf.columns)
                    tempDF.name = rchop(str, ".csv")
                    finalDf = finalDf.append(tempDF)
            elif "sex-race" in df.columns:
                tempdf = df.pivot_table(index=['sex-race'], aggfunc='size')
                # print(len(df))
                # print()
                # print(tempdf)
                # print(tempdf[0])
                for j in range(tempdf.size):
                    finalList[0] = "sex-race"
                    finalList[1] = tempdf.index[j]
                    finalList[2] = tempdf[j]
                    finalList[3] = tempdf[j] / len(df)
                    tempDF = pandas.Series(finalList, index=finalDf.columns)
                    tempDF.name = rchop(str, ".csv")
                    finalDf = finalDf.append(tempDF)
            elif "race-sex" in df.columns:
                tempdf = df.pivot_table(index=['race-sex'], aggfunc='size')
                #print(tempdf)
                #print(tempdf[0])
                for j in range(tempdf.size):
                    finalList[0]="race-sex"
                    finalList[1]=tempdf.index[j]
                    finalList[2]=tempdf[j]
                    finalList[3]=tempdf[j]/len(df)
                    tempDF = pandas.Series(finalList, index=finalDf.columns)
                    tempDF.name = rchop(str, ".csv")
                    finalDf = finalDf.append(tempDF)
            elif "sex" in df.columns and "age" in df.columns:
                df['age']=df['age'].astype(int)
                df.loc[df['age'] >=25, 'age'] = 25
                df.loc[df['age'] <25, 'age'] = 1

                df.loc[df['age'] != 25, 'age'] = "youth"
                df.loc[df['age'] == 25, 'age'] = "adult"
                if 1 in df['sex'].unique():
                    df.loc[df['sex'] != 1, 'sex'] = 'female'
                    df.loc[df['sex'] == 1, 'sex'] = 'male'

                tempdf = df.pivot_table(index=['sex','age'], aggfunc='size')
                for j in range(tempdf.size):
                    finalList[0]="sex/age"
                    finalList[1]=tempdf.index[j]
                    finalList[2]=tempdf[j]
                    finalList[3]=tempdf[j]/len(df)
                    tempDF = pandas.Series(finalList, index=finalDf.columns)
                    tempDF.name = rchop(str, ".csv")
                    finalDf = finalDf.append(tempDF)
            elif "Race" in df.columns:
                tempdf = df.pivot_table(index=['Race'], aggfunc='size')
                #print(tempdf)
                #print(tempdf[0])
                for j in range(tempdf.size):
                    finalList[0]="race"
                    finalList[1]=tempdf.index[j]
                    finalList[2]=tempdf[j]
                    finalList[3]=tempdf[j]/len(df)
                    tempDF = pandas.Series(finalList, index=finalDf.columns)
                    tempDF.name = rchop(str, ".csv")
                    finalDf = finalDf.append(tempDF)
            elif "sensitive-attr" in df.columns:
                tempdf = df.pivot_table(index=['sensitive-attr'], aggfunc='size')
                #print(tempdf)
                #print(tempdf.index[0])
                #print(tempdf[0])
                for j in range(tempdf.size):
                    finalList[0]="sensitive-attr"
                    finalList[1]=tempdf.index[j]
                    finalList[2]=tempdf[j]
                    finalList[3]=tempdf[j]/len(df)
                    tempDF = pandas.Series(finalList, index=finalDf.columns)
                    tempDF.name = rchop(str, ".csv")
                    finalDf = finalDf.append(tempDF)
            else:
                print("////////////////////////////////////////////////////////////////")
                print("----------------------------BIG PROBLEM" +str+"-------------------------")
                print("////////////////////////////////////////////////////////////////")
            finalDf = finalDf.append(pandas.Series(name=''))

        except pandas.errors.EmptyDataError:
            print("OOOPPS")

    print("_______________________________________________________")
    print(lst)
    print("_______________________________________________________")
    print(finalDf)
    finalDf.to_csv('/home/xavier/Desktop/Unbalance.csv')





main()