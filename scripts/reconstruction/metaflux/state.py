class State:
    """A dynamic object to store and pass state between pipeline tasks."""
    def __init__(self, **kwargs):
        """
        Initialize the state with optional initial values.
        :param kwargs: Initial values for the state.
        """
        self._data = kwargs

    def __getattr__(self, name):
        """Access state attributes dynamically."""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'State' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Set state attributes dynamically."""
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key):
        """Dictionary-like access for state."""
        return self._data[key]

    def __setitem__(self, key, value):
        """Dictionary-like setting for state."""
        self._data[key] = value

    def __contains__(self, key):
        """Support 'in' operator."""
        return key in self._data

    def __repr__(self):
        """Human-readable representation of the state."""
        return f"State({self._data})"

    def set(self, key, value):
        """Explicitly set a value in the state."""
        self._data[key] = value

    def get(self, key, default=None):
        """Explicitly get a value from the state with a default."""
        return self._data.get(key, default)

    def to_dict(self):
        """Convert the state to a serializable dictionary."""
        return self._data

    @staticmethod
    def from_dict(data):
        """Create a State object from a dictionary."""
        return State(**data)
