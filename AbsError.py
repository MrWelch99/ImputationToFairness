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
    lst = glob.glob("/home/xavier/.fairness/results/*.csv")
    lst=sorted(lst)
    categories = ['accuracy', 'TNR', 'TPR', 'BCR', 'CV', 'DIbinary', 'DIavgall', '0-accuracy', '1-accuracy']
    finalDf = pandas.DataFrame(None, columns=['accuracy', 'TNR', 'TPR', 'BCR', 'CV', 'DIbinary', 'DIavgall', '0-accuracy','1-accuracy'])
    for i in range(len(lst)):
        str = remove_prefix(lst[i], '/home/xavier/.fairness/results/')
        if os.path.isfile("/home/xavier/Desktop/Fairness/fairness/results/" + str):

            #print(str)
            try:
                df = pandas.read_csv('/home/xavier/.fairness/results/'+str)
                df2 = pandas.read_csv('/home/xavier/Desktop/Fairness/fairness/results/'+str)
                #print(df)

                size = len(df)
                if len(df2) < size:
                    size = len(df2)
                finalList = [None]*9
                count = 0
                if size>0:
                    for i in categories:
                        if i in df.columns and i in df2.columns:
                            lst1 = df[i].values.tolist()
                            lst2 = df2[i].values.tolist()
                            lst3 = [None]*(size)
                            for j in range(size):
                                lst3[j]=lst1[j]-lst2[j]
                            absLst3=[abs(ele) for ele in lst3]
                            finalList[count]=sum(absLst3)/len(lst3)
                        count=count+1
                    tempDF = pandas.Series(finalList,index=finalDf.columns)
                    tempDF.name=rchop(str,".csv")
                    finalDf = finalDf.append(tempDF)
                else:
                    print("No rows")
            except pandas.errors.EmptyDataError:
                print("OOOPPS")

    print("_______________________________________________________")
    print(finalList)
    print("_______________________________________________________")
    print(finalDf)
    finalDf.to_csv('/home/xavier/Desktop/AbsError.csv')





main()