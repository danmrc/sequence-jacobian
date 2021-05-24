"""Various classes to support the computation of steady states"""

from copy import deepcopy

from ..utilities.misc import dict_diff


class SteadyStateDict:
    def __init__(self, data, internal=None):
        self.toplevel = {}
        self.internal = {}
        self.update(data, internal_namespaces=internal)

    def __repr__(self):
        if self.internal:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}, internal={list(self.internal.keys())}>"
        else:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}>"

    def __iter__(self):
        return iter(self.toplevel)

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.toplevel[k]
        else:
            try:
                return {ki: self.toplevel[ki] for ki in k}
            except TypeError:
                raise TypeError(f'Key {k} needs to be a string or an iterable (list, set, etc) of strings')

    def __setitem__(self, k, v):
        self.toplevel[k] = v

    def keys(self):
        return self.toplevel.keys()

    def values(self):
        return self.toplevel.values()

    def items(self):
        return self.toplevel.items()

    def update(self, data, internal_namespaces=None):
        if isinstance(data, SteadyStateDict):
            self.internal.update(deepcopy(data.internal))
            self.toplevel.update(deepcopy(data.toplevel))
        else:
            toplevel = deepcopy(data)
            if internal_namespaces is not None:
                # Construct the internal namespace from the Block object, if a Block is provided
                if hasattr(internal_namespaces, "internal"):
                    internal_namespaces = {internal_namespaces.name: {k: v for k, v in deepcopy(data).items() if k in
                                                                      internal_namespaces.internal}}

                # Remove the internal data from `data` if it's there
                for internal_dict in internal_namespaces.values():
                    toplevel = dict_diff(toplevel, internal_dict)

                self.toplevel.update(toplevel)
                self.internal.update(internal_namespaces)
            else:
                self.toplevel.update(toplevel)

    def difference(self, data_to_remove):
        return SteadyStateDict(dict_diff(self.toplevel, data_to_remove), internal=deepcopy(self.internal))