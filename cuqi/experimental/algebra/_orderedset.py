class _OrderedSet:
    """A set (i.e. unique elements) that keeps its elements in the order they were added.

    This is a minimal implementation of an ordered set, using a dictionary for storage.
    """
    
    def __init__(self, iterable=None):
        """Initialize the OrderedSet. 

        If an iterable is provided, add all its elements to the set.
        """
        self.dict = dict.fromkeys(iterable if iterable else [])

    def add(self, item):
        """Add an item to the set. 

        If the item is already in the set, it does nothing.
        Otherwise, the item is stored as a key in the dictionary, with None as its value.
        """
        self.dict[item] = None

    def __contains__(self, item):
        """Check if an item is in the set. 

        This is equivalent to checking if the item is a key in the dictionary.
        """
        return item in self.dict

    def __iter__(self):
        """Return an iterator over the set. 

        This iterates over the keys in the dictionary.
        """
        return iter(self.dict)

    def __len__(self):
        """Return the number of items in the set."""
        return len(self.dict)

    def extend(self, other):
        """Extend the set with the items in another set.

        Raises a TypeError if the other object is not an _OrderedSet.
        """
        if not isinstance(other, _OrderedSet):
            raise TypeError("unsupported operand type(s) for extend: '_OrderedSet' and '{}'".format(type(other).__name__))
        for item in other:
            self.add(item)

    def __or__(self, other):
        """Return a new set that is the union of this set and another set.

        Raises a TypeError if the other object is not an _OrderedSet.
        """
        if not isinstance(other, _OrderedSet):
            raise TypeError("unsupported operand type(s) for |: '_OrderedSet' and '{}'".format(type(other).__name__))
        new_set = _OrderedSet(self.dict.keys())
        new_set.extend(other)
        return new_set
