from itertools import chain


def h5group_to_dict(h5group):
    group_dict = {k: v[()] for k, v in chain(h5group.items(), h5group.attrs.items())}
    return group_dict
