class Op(object):
    """
    Base class
    all operators have a single parent
    an operator may have multiple children
    """
    _id = 0
    def __init__(self):
        self.p = None
        self.id = Op._id
        Op._id += 1
    def replace(self, newop):
        """
        Replace myself with @newop in the tree.
        The key is to reassign the parent and child pointers appropriately.
        """
        if not self.p: return
        p = self.p
        newop.p = p
        if isinstance(p, UnaryOp):
            p.c = newop
        if isinstance(p, BinaryOp):
            if p.l == self:
                p.l = newop
            elif p.r == self:
                p.r = newop
        if isinstance(p, NaryOp):
            if self in p.cs:
                cs[cs.index(self)] = newop
            p.cs = cs
    def is_ancestor(self, anc):
        """
        Check if @anc is an ancestor of the current operator
        """
        n = self
        seen = set()
        while n and n not in seen:
            seen.add(n)
            if n == anc:
                return True
            n = n.p
        return False
    def children(self):
        """
        return child operators that are relational operations (as opposed to Expressions)
        """
        children = []
        if self.is_type(UnaryOp):
            children = [self.c]
        if self.is_type(BinaryOp):
            children = [self.l, self.r]
        if self.is_type(NaryOp):
            children = list(self.cs)
        return filter(bool, children)
    def referenced_op_children(self):
        """
        return all Op subclasses referenced by current operator
        """
        children = []
        for key, attrval in self.__dict__.items():
            if key in ["p"]:   # avoid following cycles
                continue
        if not isinstance(attrval, list):
            attrval = [attrval]
        for v in attrval:
            if v and isinstance(v, Op):
                  children.append(v)
        return children
    def traverse(self, f, path=None):
        """
        Visit all referenced Op subclasses and call f()
        @f a function that takes as input the current operator and 
        the path to the operator.  f() can return False to
        stop traversing subplans.
        """
        if path is None:
            path = []
        path = path + [self]
        if f(self, path) == False:
            return
        for child in self.referenced_op_children():
            child.traverse(f, path)
    def is_type(self, klass_or_names):
        """
        Check whether self is a subclass of argument
        @klass_or_names an individual or list of classes
        """
        if not isinstance(klass_or_names, list):
            klass_or_names = [klass_or_names]
        names = [kn for kn in klass_or_names if isinstance(kn, str)]
        klasses = [kn for kn in klass_or_names if isinstance(kn, type)]
        return (self.__class__.__name__ in names or
           any([isinstance(self, kn) for kn in klasses]))
    def collect(self, klass_or_names):
        """
        Returns all operators in the subplan rooted at the current object
        that has the same class name, or is a subclass, as the arguments
        """
        ret = []
        if not isinstance(klass_or_names, list):
            klass_or_names = [klass_or_names]
        names = [kn for kn in klass_or_names if isinstance(kn, str)]
        klasses = [kn for kn in klass_or_names if isinstance(kn, type)]

        def f(node, path):
            if node and (
              node.__class__.__name__ in names or
              any([isinstance(node, kn) for kn in klasses])):
                ret.append(node)
        self.traverse(f)
        return ret
    def collectone(self, klassnames):
        """
        Helper function to return an arbitrary operator that matches any of the
        klass names or klass objects, or None
        """
        l = self.collect(klassnames)
        if l:
            return l[0]
        return None
    
class UnaryOp(Op):
    def __init__(self, c=None):
        super()
        self.c = c
        if c:
            c.p = self

            
class BinaryOp(Op):
    def __init__(self, l, r):
        super()
        self.l = l
        self.r = r
        if l:
            l.p = self
        if r:
            r.p = self
            
class NaryOp(Op):
    def __init__(self, cs):
        super()
        self.cs = cs
        for c in cs:
            if c:
                c.p = self

