def field_format(name: str, fs: str | None):
    if fs:
        return "{%s:%s}" % (name, fs)
    else:
        return "{%s}" % (name,)
