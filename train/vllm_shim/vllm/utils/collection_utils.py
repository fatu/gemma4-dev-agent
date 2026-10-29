def is_list_of(value, typ, *, check="first"):
    if not isinstance(value, list):
        return False
    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    return all(isinstance(v, typ) for v in value)
