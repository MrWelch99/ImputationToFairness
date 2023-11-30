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
    lst = glob.glob("/home/xavier/Desktop/Investigação-Fairness/FairnessADASYN/results-100/*.csv")
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





main()