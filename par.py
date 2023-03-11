from multiprocessing import cpu_count, Pool
import numpy as np

def mathblast(x, size):
    return np.tile(x, size)


def main():
    arr = np.zeros((10000, 1))

    pool = Pool(cpu_count())

    results = []
    for i in range(5):
        r = pool.apply_async(mathblast, args=(i, (2000, 1)))
        results.append(r)

    pool.close()
    pool.join()

    for i, r in enumerate(results):
        try:
            arr[i*2000: i*2000+2000] = r.get()
        except Exception:
            raise Exception("failed")

    a=3



if __name__ == '__main__':
    main()