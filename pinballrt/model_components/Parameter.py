
class Parameter:
    def __init__(self, name, value, kind, prior_range=None, fixed=True, 
                 unit=None, log=False):
        self.name = name
        self.value = value
        self.prior_range = prior_range
        self.fixed = fixed
        self.unit = unit
        self.log=log
        self.kind = kind

    @property
    def quantity(self):
        if isinstance(self.value, str):
            return self.value
        elif isinstance(self.value, bool):
            return self.value
        else:
            physical = 10**self.value if self.log else self.value
            return physical * self.unit if self.unit is not None else physical
