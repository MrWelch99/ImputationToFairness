import multiprocessing as mp
import psutil 


def my_func(x):
    print(x**x)

def main():
    print("MULTIPROCESSING CPU COUNT: "+str(mp.cpu_count()))
    print("PSUTIL CPU COUNT: "+str(psutil.cpu_count(logical = False)))
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_func, [4,2,3])

if __name__ == "__main__":
    main()