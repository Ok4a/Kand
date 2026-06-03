import numpy as np
import multiprocessing as mp

ebutton = mp.Event()


def sq(x,y):
    np.seterr(over='raise')

    try:
        print(ebutton.is_set())
        if ebutton.is_set():
            return -1
        if y == 0:
            print(x)
            np.int16(32000) * np.int16(x)
            return x
        else:
            return x

    except FloatingPointError:
        ebutton.set()
        return -2




if __name__ == '__main__':

    # orig = np.seterr(over='call')
    # print(orig)

    # print(sq(1,1))

    for i in range(2):

        pool = mp.Pool()
        pool_test1 = pool.starmap(sq, [(x,i) for x in range(10)])
        print(pool_test1)


        
        # with mp.Pool() as pool:

        # pool_test2 = pool.starmap(sq, [(x,0) for x in range(10)])

    pool.close()
    pool.join()
