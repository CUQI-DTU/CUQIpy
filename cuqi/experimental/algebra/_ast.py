"""
CUQIpy specific implementation of an abstract syntax tree (AST) for algebra on variables.

The AST is used to record the operations applied to variables allowing a delayed evaluation
of said operations when needed by traversing the tree with the __call__ method.

For example, the following code

    x = VariableNode('x')
    y = VariableNode('y')
    z = 2*x + 3*y

will create the following AST:

z = AddNode(
        MultiplyNode(
            ValueNode(2),
            VariableNode('x')
        ),
        MultiplyNode(
            ValueNode(3),
            VariableNode('y')
        )
    )

which can be evaluated by calling the __call__ method:

    z(x=1, y=2) # returns 8

"""

from abc import ABC, abstractmethod

convert_to_node = lambda x: x if isinstance(x, Node) else ValueNode(x)
""" Converts any non-Node object to a ValueNode object. """

# ====== Base classes for the nodes ======


class Node(ABC):
    """Base class for all nodes in the abstract syntax tree.

    Responsible for building the AST by creating nodes that represent the operations applied to variables.

    Each subclass must implement the __call__ method that will evaluate the node given the input parameters.

    """

    @abstractmethod
    def __call__(self, **kwargs):
        """Evaluate node at a given parameter value. This will traverse the sub-tree originated at this node and evaluate it given the recorded operations."""
        pass

    @abstractmethod
    def condition(self, **kwargs):
        """ Conditions the tree by replacing any VariableNode with a ValueNode if the variable is in the kwargs dictionary. """
        pass

    @abstractmethod
    def __repr__(self):
        """String representation of the node. Used for printing the AST."""
        pass

    def get_variables(self, variables=None):
        """Returns a set with the names of all variables in the sub-tree originated at this node."""
        if variables is None:
            variables = set()
        if isinstance(self, VariableNode):
            variables.add(self.name)
        if hasattr(self, "child"):
            self.child.get_variables(variables)
        if hasattr(self, "left"):
            self.left.get_variables(variables)
        if hasattr(self, "right"):
            self.right.get_variables(variables)
        return variables

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


class UnaryNode(Node, ABC):
    """Base class for all unary nodes in the abstract syntax tree.

    Parameters
    ----------
    child : Node
        The direct child node on which the unary operation is performed.

    """

    def __init__(self, child: Node):
        self.child = child

    def condition(self, **kwargs):
        return self.__class__(self.child.condition(**kwargs))


class BinaryNode(Node, ABC):
    """Base class for all binary nodes in the abstract syntax tree.

    The op_symbol attribute is used for printing the operation in the __repr__ method.

    Parameters
    ----------
    left : Node
        Left child node to the binary operation.

    right : Node
        Right child node to the binary operation.

    """

    @property
    @abstractmethod
    def op_symbol(self):
        """Symbol used to represent the operation in the __repr__ method."""
        pass

    def __init__(self, left: Node, right: Node):
        self.left = left
        self.right = right

    def condition(self, **kwargs):
        return self.__class__(self.left.condition(**kwargs), self.right.condition(**kwargs))

    def __repr__(self):
        return f"{self.left} {self.op_symbol} {self.right}"


class BinaryNodeWithParenthesis(BinaryNode, ABC):
    """Base class for all binary nodes in the abstract syntax tree that should be printed with parenthesis."""

    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        right = (
            f"({self.right})" if isinstance(self.right, BinaryNode) else str(self.right)
        )
        return f"{left} {self.op_symbol} {right}"

class BinaryNodeWithParenthesisNoSpace(BinaryNode, ABC):
    """Base class for all binary nodes in the abstract syntax tree that should be printed with parenthesis but no space."""

    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        right = (
            f"({self.right})" if isinstance(self.right, BinaryNode) else str(self.right)
        )
        return f"{left}{self.op_symbol}{right}"


# ====== Specific implementations of the "leaf" nodes ======


class VariableNode(Node):
    """Node that represents a generic variable, e.g. "x" or "y".

    Parameters
    ----------
    name : str
        Name of the variable. Used for printing and to retrieve the given input value
        of the variable in the kwargs dictionary when evaluating the tree.

    """

    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        """Retrieves the value of the variable from the passed kwargs. If no value is found, it raises a KeyError."""
        if not self.name in kwargs:
            raise KeyError(
                f"Variable '{self.name}' not found in the given input parameters. Unable to evaluate the expression."
            )
        return kwargs[self.name]

    def condition(self, **kwargs):
        if self.name in kwargs:
            return ValueNode(kwargs[self.name])
        return self

    def __repr__(self):
        return self.name


class ValueNode(Node):
    """Node that represents a constant value. The value can be any python object that is not a Node.

    Parameters
    ----------
    value : object
        The python object that represents the value of the node.

    """

    def __init__(self, value):
        self.value = value

    def __call__(self, **kwargs):
        """Returns the value of the node."""
        return self.value

    def condition(self, **kwargs):
        return self

    def __repr__(self):
        return str(self.value)


# ====== Specific implementations of the "internal" nodes ======


class AddNode(BinaryNode):
    """Node that represents the addition operation."""

    @property
    def op_symbol(self):
        return "+"

    def __call__(self, **kwargs):
        return self.left(**kwargs) + self.right(**kwargs)


class SubtractNode(BinaryNode):
    """Node that represents the subtraction operation."""

    @property
    def op_symbol(self):
        return "-"

    def __call__(self, **kwargs):
        return self.left(**kwargs) - self.right(**kwargs)


class MultiplyNode(BinaryNodeWithParenthesis):
    """Node that represents the multiplication operation."""

    @property
    def op_symbol(self):
        return "*"

    def __call__(self, **kwargs):
        return self.left(**kwargs) * self.right(**kwargs)


class DivideNode(BinaryNodeWithParenthesis):
    """Node that represents the division operation."""

    @property
    def op_symbol(self):
        return "/"

    def __call__(self, **kwargs):
        return self.left(**kwargs) / self.right(**kwargs)


class PowerNode(BinaryNodeWithParenthesisNoSpace):
    """Node that represents the power operation."""

    @property
    def op_symbol(self):
        return "^"

    def __call__(self, **kwargs):
        return self.left(**kwargs) ** self.right(**kwargs)


class GetItemNode(BinaryNode):
    """Node that represents the get item operation. Here the left node is the object and the right node is the index."""

    def __call__(self, **kwargs):
        return self.left(**kwargs)[self.right(**kwargs)]

    def __repr__(self):
        left = f"({self.left})" if isinstance(self.left, BinaryNode) else str(self.left)
        return f"{left}[{self.right}]"
    
    @property
    def op_symbol(self):
        pass


class NegateNode(UnaryNode):
    """Node that represents the arithmetic negation operation."""

    def __call__(self, **kwargs):
        return -self.child(**kwargs)

    def __repr__(self):
        child = (
            f"({self.child})"
            if isinstance(self.child, (BinaryNode, UnaryNode))
            else str(self.child)
        )
        return f"-{child}"


class AbsNode(UnaryNode):
    """Node that represents the absolute value operation."""

    def __call__(self, **kwargs):
        return abs(self.child(**kwargs))

    def __repr__(self):
        return f"abs({self.child})"


class MatMulNode(BinaryNodeWithParenthesis):
    """Node that represents the matrix multiplication operation."""

    @property
    def op_symbol(self):
        return "@"

    def __call__(self, **kwargs):
        return self.left(**kwargs) @ self.right(**kwargs)
