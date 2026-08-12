import copy
import warnings
import numpy as np


class ParameterList:

    def __init__(self, params=None):
        """
        params: iterable of Parameter objects, each with a .name attribute.
                Example: ParameterList([param1, param2])
        """
        self._params = {}
        if params:
            for p in params:
                self.add(p)

    def add(self, param):
        self._params[(param.name, param.kind)] = param

    def __getitem__(self, key):
        if isinstance(key, str):
            matches = [p for (name, kind), p in self._params.items() if name == key]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise KeyError(f"Ambiguous: multiple parameters named '{key}' with different kinds. Use a (name, kind) tuple to index.")
            raise KeyError(key)
        elif isinstance(key, tuple):
            return self._params[key]
        elif isinstance(key, int):
            return list(self._params.values())[key]
        elif isinstance(key, slice):
            result = ParameterList()
            for param in list(self._params.values())[key]:
                result.add(param)
            return result
        raise TypeError(f"indices must be str, tuple, int, or slice, not {type(key).__name__}")

    def __contains__(self, key):
        if isinstance(key, tuple):
            return key in self._params
        return any(name == key for name, kind in self._params)

    def __iter__(self):
        return iter(self._params.values())

    def __len__(self):
        return len(self._params)

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    def items(self):
        return self._params.items()

    @property
    def free(self):
        return [p for p in self._params.values() if not p.fixed]

    @property
    def vector(self):
        return np.array([p.value for p in self.free])

    def set_vector(self, v):
        for param, val in zip(self.free, v):
            param.value = val

    def by_kind(self, kind):
        result = ParameterList()
        for param in self._params.values():
            if param.kind == kind:
                result.add(param)
        return result

    def copy(self):
        result = ParameterList()
        for param in self._params.values():
            result.add(copy.deepcopy(param))
        return result

    def __add__(self, other):
        conflicts = self._params.keys() & other._params.keys()
        if conflicts:
            names = ", ".join(f"'{n}' (kind='{k}')" for n, k in conflicts)
            warnings.warn(f"Duplicate parameters kept from first ParameterList: {names}", UserWarning)
        result = ParameterList()
        for param in self._params.values():
            result.add(param)
        for param in other._params.values():
            if (param.name, param.kind) not in result._params:
                result.add(param)
        return result

    def __repr__(self):
        if not self._params:
            return "ParameterList()"
        lines = []
        for p in self._params.values():
            fixed_str = "fixed" if p.fixed else "free"
            lines.append(f"  {p.name}: {p.quantity} [{fixed_str}, {p.kind}]")
        return "ParameterList(\n" + "\n".join(lines) + "\n)"
