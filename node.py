class Node(object):
  def __init__(self):
    # Node(s) from which this Node receives values
    self.inbound_nodes = inbound_nodes
    # Node(s) to which this Node passes values
    self.outbound_nodes = []
    # For each inbound Node here, add this Node as an outbound to that Node.
    for n in self.inbound_nodes:
      n.outbound_nodes.append(self)

    # A calculated value
    self.value = None
