convert_to_node = lambda x: x if isinstance(x, Node) else ValueNode(x)

class Node:
    def __call__(self, **kwargs):
        raise NotImplementedError()

    def __add__(self, other):
        return AddNode(self, convert_to_node(other))
    
    def __radd__(self, other):
        return AddNode(convert_to_node(other), self)

    def __sub__(self, other):
        return SubtractNode(self, convert_to_node(other))
    
    def __rsub__(self, other):
        return SubtractNode(convert_to_node(other), self)

    def __mul__(self, other):
        return MultiplyNode(self, convert_to_node(other))
    
    def __rmul__(self, other):
        return MultiplyNode(convert_to_node(other), self)

    def __truediv__(self, other):
        return DivideNode(self, convert_to_node(other))
    
    def __rtruediv__(self, other):
        return DivideNode(convert_to_node(other), self)

    def __pow__(self, other):
        return PowerNode(self, convert_to_node(other))
    
    def __rpow__(self, other):
        return PowerNode(convert_to_node(other), self)

    def __neg__(self):
        return NegateNode(self)

    def __abs__(self):
        return AbsNode(self)

    def __getitem__(self, i):
        return GetItemNode(self, convert_to_node(i))
    
    def __matmul__(self, other):
        return MatMulNode(self, convert_to_node(other))
    
    def __rmatmul__(self, other):
        return MatMulNode(convert_to_node(other), self)    

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


class BinaryNode(Node):
    op = None
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.left} {self.op} {self.right}"
    
class BinaryNodeWithParenthesis(BinaryNode):
    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        right = f"({self.right})" if isinstance(self.right, BinaryNode) else str(self.right)
        return f"{left} {self.op} {right}"

class UnaryNode(Node):
    def __init__(self, child: Node):
        self.child = child

    def __repr__(self):
        raise NotImplementedError()

class AddNode(BinaryNode):
    op = '+'
    def __call__(self, **kwargs):
        return self.left(**kwargs) + self.right(**kwargs)

class SubtractNode(BinaryNode):
    op = '-'
    def __call__(self, **kwargs):
        return self.left(**kwargs) - self.right(**kwargs)

class MultiplyNode(BinaryNodeWithParenthesis):
    op = '*'
    def __call__(self, **kwargs):
        return self.left(**kwargs) * self.right(**kwargs)

class DivideNode(BinaryNodeWithParenthesis):
    op = '/'
    def __call__(self, **kwargs):
        return self.left(**kwargs) / self.right(**kwargs)

class PowerNode(BinaryNodeWithParenthesis):
    op = '^'
    def __call__(self, **kwargs):
        return self.left(**kwargs) ** self.right(**kwargs)

class GetItemNode(BinaryNode):
    def __call__(self, **kwargs):
        return self.left(**kwargs)[self.right(**kwargs)]

    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        return f"{left}[{self.right}]"

class NegateNode(UnaryNode):
    def __call__(self, **kwargs):
        return -self.child(**kwargs)
    
    def __repr__(self):
        child = f"({self.child})" if isinstance(self.child, (BinaryNode, UnaryNode)) else str(self.child)
        return f"-{child}"

class AbsNode(UnaryNode):
    def __call__(self, **kwargs):
        return abs(self.child(**kwargs))

    def __repr__(self):
        return f"abs({self.child})"
    
class MatMulNode(BinaryNodeWithParenthesis):
    op = '@'
    def __call__(self, **kwargs):
        return self.left(**kwargs) @ self.right(**kwargs)
