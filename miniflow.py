class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an outbound to that Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and store the result
        in self.value.
        """
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes, so no need to pass anything to the
        # Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value may be passed as an
    # argument to forward().
    #
    # All other node implementations should get the value of the previous node
    # from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self, *inputs):
        # Add class takes 2 inbound nodes, x and y, and adds the values of those
        # nodes.
        Node.__init__(self, inputs)

    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of it's inbound_nodes.

        Your code here!
        """
        self.value = 0
        for n in self.inbound_nodes:
            self.value += n.value
