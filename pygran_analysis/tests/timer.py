from timeit import Timer


def timeit(func, *args, **kwargs):
    output_container = []
    iters = kwargs.pop("iters") if "iters" in kwargs else 100

    def wrapper():
        output_container.append(func(*args, **kwargs))

    timer = Timer(wrapper)
    delta = timer.timeit(number=iters)
    return delta, output_container.pop()
