import logging
import time
from multiprocessing.pool import ThreadPool
from typing import Union, List, Any


def timeit(logger: logging.Logger = None):
    def wrap(f):
        def wrapped_f(*args, **kwargs):
            ts = time.time()
            result = f(*args, **kwargs)
            te = time.time()
            if logger:
                logger.debug('function: {} took: {} sec'.format(f.__name__, te - ts))
            return result

        return wrapped_f

    return wrap


@timeit()
def run_parallel(func, iterable) -> Union[None, List[Any]]:
    _list_res = None
    try:
        with ThreadPool() as pool:
            _list_res = pool.starmap(func=func, iterable=iterable)
    except Exception as e:
        print(e)
    return _list_res


if __name__ == '__main__':
    @timeit
    def f(a: str, b: str) -> str:
        time.sleep(1)
        return "{} and {}".format(a, b)


    list_names = [('BMW', 'VW'), ('Porsche', 'Lada')]
    list_res = run_parallel(func=f, iterable=list_names)
