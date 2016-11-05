def print_args(func):
    def func_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        to_str = lambda x: x.__name__ if callable(x) else str(x)
        print("Parameters: ", ", ".join(
                [str(x) + "=" + str(y) for x, y in zip(func.__code__.co_varnames[2:len(args)], args[2:])] + 
                [str(key) + "=" + to_str(value) for key, value in kwargs.items()]))
        print("Result: ", str(result))
        print("-"*50)
        return result
    return func_wrapper