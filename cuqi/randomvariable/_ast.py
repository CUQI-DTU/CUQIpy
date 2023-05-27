convert_to_node = lambda x: x if isinstance(x, Node) else ValueNode(x)

class Node:
    def __call__(self, **kwargs):
        raise NotImplementedError()
    
    def __add__(self, other):
        return AddNode(self, convert_to_node(other))
    
    def __mul__(self, other):
        return MultiplyNode(self, convert_to_node(other))
    
class RandomVariableNode(Node):
    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        return kwargs.get(self.name, 0)

    def __repr__(self):
        return self.name

class ValueNode(Node):
    def __init__(self, value):
        self.value = value

    def __call__(self, **kwargs):
        return self.value

    def __repr__(self):
        return str(self.value)
    
    
class AddNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) + self.right(**kwargs)

    def __repr__(self):
        return f"{self.left} + {self.right}"
    
class MultiplyNode(Node):
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __call__(self, **kwargs):
        return self.left(**kwargs) * self.right(**kwargs)

    def __repr__(self):
        return f"( {self.left} ) * ( {self.right} )"
