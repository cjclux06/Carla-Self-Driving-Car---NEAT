import random

class Connection:

    def __init__(self, in_node_id=None, out_node_id=None, weight=None, expressed=None, innovation=None, connection=None):
        if connection is not None:
            # Copy from existing connection
            self.in_node_id = connection.in_node_id
            self.out_node_id = connection.out_node_id
            self.weight = connection.weight
            self.expressed = connection.expressed
            self.innovation = connection.innovation
        else:
            # Create new connection - ensure no None values
            self.in_node_id = in_node_id
            self.out_node_id = out_node_id
            self.weight = weight if weight is not None else random.random() * 2 - 1  # DEFAULT WEIGHT
            self.expressed = expressed if expressed is not None else True  # DEFAULT EXPRESSED
            self.innovation = innovation