def state_name_gen(prefix='s'):
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1