"""
CUQIpy specific implementation of an abstract syntax tree (AST) for random variables.

The main purpose of this AST is to allow for a simple and intuitive way of implementing
algebraic operations on random variables. The AST is used to record the operations
and then evaluate them when needed by traversing the tree with the __call__ method.

For example, the following code

    x = RandomVariableNode('x')
    y = RandomVariableNode('y')
    z = 2*x + 3*y

will create the following AST:

    AddNode(
        MultiplyNode(
            ValueNode(2),
            RandomVariableNode('x')
        ),
        MultiplyNode(
            ValueNode(3),
            RandomVariableNode('y')
        )
    )

which can be evaluated by calling the __call__ method:

    z(x=1, y=2) # returns 8


"""
convert_to_node = lambda x: x if isinstance(x, Node) else ValueNode(x)

class Node:
    """ Base class for all nodes in the abstract syntax tree.  """
    def __call__(self, **kwargs):
        """ Evaluate node at a given parameter value. This will traverse the tree and evaluate given the recorded operations. """
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
    """ Node that represents a random variable.
    
    Parameters
    ----------
    name : str
        Name of the random variable. Used for printing and to retrieve the given input value
        of the random variable in the kwargs dictionary when evaluating the tree. 
        
    """
    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        return kwargs.get(self.name, 0)

    def __repr__(self):
        return self.name

class ValueNode(Node):
    """ Node that represents a constant value. The value can be any python object.
    
    Parameters
    ----------
    value : object
        Value of the node.
        
    """
    def __init__(self, value):
        self.value = value

    def __call__(self, **kwargs):
        return self.value

    def __repr__(self):
        return str(self.value)

class BinaryNode(Node):
    """ Base class for all binary nodes in the abstract syntax tree.
    
    The op_symbol attribute is used for printing the operation in the __repr__ method.
    
    Parameters
    ----------
    left : Node
        Left child node to the binary operation.
        
    right : Node
        Right child node to the binary operation.
        
    """
    op_symbol = None
    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"{self.left} {self.op_symbol} {self.right}"
    
class BinaryNodeWithParenthesis(BinaryNode):
    """ Base class for all binary nodes in the abstract syntax tree that should be printed with parenthesis."""
    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        right = f"({self.right})" if isinstance(self.right, BinaryNode) else str(self.right)
        return f"{left} {self.op_symbol} {right}"

class UnaryNode(Node):
    """ Base class for all unary nodes in the abstract syntax tree.

    Parameters
    ----------
    child : Node
        The direct child node on which the unary operation is performed.

    """
    def __init__(self, child: Node):
        self.child = child

    def __repr__(self):
        raise NotImplementedError()

class AddNode(BinaryNode):
    """ Node that represents the addition operation."""
    op_symbol = '+'
    def __call__(self, **kwargs):
        return self.left(**kwargs) + self.right(**kwargs)

class SubtractNode(BinaryNode):
    """ Node that represents the subtraction operation."""
    op_symbol = '-'
    def __call__(self, **kwargs):
        return self.left(**kwargs) - self.right(**kwargs)

class MultiplyNode(BinaryNodeWithParenthesis):
    """ Node that represents the multiplication operation."""
    op_symbol = '*'
    def __call__(self, **kwargs):
        return self.left(**kwargs) * self.right(**kwargs)

class DivideNode(BinaryNodeWithParenthesis):
    """ Node that represents the division operation."""
    op_symbol = '/'
    def __call__(self, **kwargs):
        return self.left(**kwargs) / self.right(**kwargs)

class PowerNode(BinaryNodeWithParenthesis):
    """ Node that represents the power operation."""
    op_symbol = '^'
    def __call__(self, **kwargs):
        return self.left(**kwargs) ** self.right(**kwargs)

class GetItemNode(BinaryNode):
    """ Node that represents the get item operation. Here the left node is the object and the right node is the index. """
    def __call__(self, **kwargs):
        return self.left(**kwargs)[self.right(**kwargs)]

    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        return f"{left}[{self.right}]"

class NegateNode(UnaryNode):
    """ Node that represents the negation operation."""
    def __call__(self, **kwargs):
        return -self.child(**kwargs)
    
    def __repr__(self):
        child = f"({self.child})" if isinstance(self.child, (BinaryNode, UnaryNode)) else str(self.child)
        return f"-{child}"

class AbsNode(UnaryNode):
    """ Node that represents the absolute value operation."""
    def __call__(self, **kwargs):
        return abs(self.child(**kwargs))

    def __repr__(self):
        return f"abs({self.child})"
    
class MatMulNode(BinaryNodeWithParenthesis):
    """ Node that represents the matrix multiplication operation."""
    op_symbol = '@'
    def __call__(self, **kwargs):
        return self.left(**kwargs) @ self.right(**kwargs)
