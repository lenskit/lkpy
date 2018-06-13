def compute(x):
    if hasattr(x, 'compute'):
        return x.compute()
    else:
        return x
