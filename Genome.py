from Connection import Connection
from Node import Node, NodeType
from Counter import Counter
import random
import copy

class Genome():

    global perturbing_rate
    perturbing_rate = 0.9

    def __init__(self, original_genome=None):
        self.nodes = dict()
        self.connections = dict()
        self.fitness = 0   

        if original_genome != None:       
            for index in original_genome.nodes.keys():
                self.nodes[index] = copy.deepcopy(original_genome.nodes.get(index))

            for index in original_genome.connections.keys():
                self.connections[index] = copy.deepcopy(original_genome.connections.get(index))

    def add_node(self, node):
        self.nodes[node.innovation] = node

    def add_connection(self, connection):
        self.connections[connection.innovation] = connection

    def mutation(self):
        mutation_count = 0
        for connection in self.connections.values():
            if random.random() < perturbing_rate:
                old_weight = connection.weight
                connection.weight += random.gauss(0, 0.1)
                if old_weight != connection.weight:
                    mutation_count += 1
            elif random.random() < 0.1:
                connection.weight = random.random() * 4 - 2
                mutation_count += 1
        #print(f"Mutations applied: {mutation_count}")  # Uncomment to see

    def connection_mutation(self, innovation, max_attempts):
        tries = 0
        success = False
        while tries < max_attempts and success == False:
            tries += 1

            node1 = random.choice(list(self.nodes.values()))
            node2 = random.choice(list(self.nodes.values()))
            weight = random.random() * 2 - 1

            is_reversed = False

            if node1.type == NodeType.HIDDEN and node2.type == NodeType.INPUT:
                is_reversed = True
            elif node1.type == NodeType.OUTPUT and node2.type == NodeType.HIDDEN:
                is_reversed = True
            elif node1.type == NodeType.OUTPUT and node2.type == NodeType.INPUT:
                is_reversed = True

            connection_impossible = False
            if node1.type == NodeType.INPUT and node2.type == NodeType.INPUT:
                connection_impossible = True
            elif node1.type == NodeType.OUTPUT and node2.type == NodeType.OUTPUT:
                connection_impossible = True

            connection_exists = False
            for connection in self.connections.values():
                if connection.in_node_id == node1.innovation and connection.out_node_id == node2.innovation:
                    connection_exists = True
                elif connection.in_node_id == node2.innovation and connection.out_node_id == node1.innovation:
                    connection_exists = True

            if connection_exists or connection_impossible:
                continue

            v1 = node2 if is_reversed else node1
            v2 = node1 if is_reversed else node2

            new_connection = Connection(v1.innovation, v2.innovation, weight, True, innovation.get_current_innovation())
            self.connections[new_connection.innovation] = new_connection
            success = True
        if success == False:
            #print("Tried, but could not add more connections")
            pass

    def node_mutation(self, connection_innovation, node_innovation):
        connection = random.choice(list(self.connections.values()))
        in_node = self.nodes[connection.in_node_id]
        out_node = self.nodes[connection.out_node_id]

        connection.expressed = False

        new_node = Node(type=NodeType.HIDDEN, innovation=node_innovation.get_current_innovation())
        in_new_node = Connection(in_node.innovation, new_node.innovation, 1, True, connection_innovation.get_current_innovation())
        out_new_node = Connection(new_node.innovation, out_node.innovation, connection.weight, True, connection_innovation.get_current_innovation())
		
        self.nodes[new_node.innovation] = new_node
        self.connections[in_new_node.innovation] = in_new_node
        self.connections[out_new_node.innovation] = out_new_node

    @staticmethod
    
    def crossover(parent1, parent2):
        child = Genome()
    
        # Copy all nodes from both parents (handling duplicates)
        all_nodes = {}
        for node in list(parent1.nodes.values()) + list(parent2.nodes.values()):
            if node.innovation not in all_nodes:
                all_nodes[node.innovation] = copy.deepcopy(node)
    
        for node in all_nodes.values():
            child.add_node(node)
    
        # Copy connections with proper matching
        all_connections = {}
    
        # Add connections from both parents
        for conn in parent1.connections.values():
            all_connections[conn.innovation] = copy.deepcopy(conn)
    
        for conn in parent2.connections.values():
            if conn.innovation in all_connections:
                # Both parents have this connection - randomly choose one
                if random.random() < 0.5:
                    all_connections[conn.innovation] = copy.deepcopy(conn)
    
        for conn in all_connections.values():
            child.add_connection(Connection(conn.in_node_id, conn.out_node_id, conn.weight, conn.expressed, conn.innovation))
    
        return child
    

    def count_disjoint_genes(self, genome1, genome2):

        disjoint_genes = 0
        excess_genes = 0

        list1 = list(genome1.nodes.keys())
        list2 = list(genome2.nodes.keys())

        innovation1 = max(list1) if list1 else 0
        innovation2 = max(list2) if list2 else 0

        highest = max(max(list1) if list1 else 0, max(list2) if list2 else 0)

        for i in range(highest + 1):
            node1 = genome1.nodes.get(i)
            node2 = genome2.nodes.get(i)
            if node1 is None and innovation1 > i and not node2 is None:
                disjoint_genes += 1
            elif node2 is None and innovation2 > i and not node1 is None:
                disjoint_genes += 1
            if node1 is None and innovation1 < i and not node2 is None:
                excess_genes += 1
            elif node2 is None and innovation2 < i and not node1 is None:
                excess_genes += 1

        list1 = list(genome1.connections.keys())
        list2 = list(genome2.connections.keys())

        innovation1 = max(list1) if list1 else 0
        innovation2 = max(list2) if list2 else 0

        highest = max(max(list1) if list1 else 0, max(list2) if list2 else 0)

        for i in range(highest + 1):
            connection1 = genome1.connections.get(i)
            connection2 = genome2.connections.get(i)
            if connection1 == None and innovation1 > i and connection2 != None:
                disjoint_genes += 1
            elif connection2 == None and innovation2 > i and connection1 != None:
                disjoint_genes += 1
            if connection1 == None and innovation1 < i and connection2 != None:
                excess_genes += 1
            elif connection2 == None and innovation2 < i and connection1 != None:
                excess_genes += 1

        return disjoint_genes, excess_genes


    def count_average_weight_difference(self, genome1, genome2):

        matching_genes = 0
        weight_difference = 0

        list1 = list(genome1.connections.keys())
        list2 = list(genome2.connections.keys())

        highest = max(max(list1) if list1 else 0, max(list2) if list2 else 0)

        for i in range(highest):
            connection1 = genome1.connections.get(i)
            connection2 = genome2.connections.get(i)
            if connection1 != None and connection2 != None:
                matching_genes += 1
                weight_difference += abs(connection1.weight - connection2.weight)
        if matching_genes == 0:
            return 0
        return weight_difference/matching_genes

    def compatibility_distance(self, genome1, genome2, c1, c2, c3):

        disjoint_genes, excess_genes = self.count_disjoint_genes(genome1, genome2)
        average_weight_difference = self.count_average_weight_difference(genome1, genome2)

        return excess_genes * c1 + disjoint_genes * c2 + average_weight_difference * c3
    
    def count_matching_genes(self, genome1, genome2):

        matching_genes = 0

        list1 = list(genome1.nodes.keys())
        list2 = list(genome2.nodes.keys())

        highest = max([len(list1) - 1, len(list2) - 1])

        for i in range(highest):
            node1 = genome1.nodes.get(i)
            node2 = genome2.nodes.get(i)
            if not node1 is None and not node2 is None:
                matching_genes += 1

        list1 = list(genome1.connections.keys())
        list2 = list(genome2.connections.keys())

        highest = max([len(list1) - 1, len(list2) - 1])

        for i in range(highest):
            connection1 = genome1.connections.get(i)
            connection2 = genome2.connections.get(i)
            if connection1 != None and connection2 != None:
                matching_genes += 1

        return matching_genes
    
    def activate_inputs(self, inputs):
        # Reset all non-input nodes
        for node in self.nodes.values():
            if node.type is NodeType.INPUT:
                node.value = 0.0

        # Assign inputs
        input_idx = 0
        for node in self.nodes.values():
            if node.type is NodeType.INPUT:
                if input_idx < len(inputs):
                    node.value = inputs[input_idx]
                    input_idx += 1

        # Topological sort so values flow correctly input -> hidden -> output
        visited = set()
        order = []

        # Iterative post-order DFS (replaces the recursive visit function)
        stack = list(self.nodes.keys())
        while stack:
            node_id = stack[-1]
            if node_id in visited:
                stack.pop()
                if node_id not in order:
                    order.append(node_id)
                continue
            visited.add(node_id)
            for conn in self.connections.values():
                if conn.expressed and conn.out_node_id == node_id and conn.in_node_id not in visited:
                    stack.append(conn.in_node_id)

        # Reset non-input node values before accumulation
        for node in self.nodes.values():
            if node.type is not NodeType.INPUT:
                node.value = 0.0

        # Propagate in topological order
        for node_id in order:
            node = self.nodes[node_id]
            for conn in self.connections.values():
                if conn.expressed and conn.in_node_id == node_id:
                    self.nodes[conn.out_node_id].value += node.value * conn.weight

            # Apply correct activation based on node type
            if node.type is NodeType.HIDDEN:      # hidden
                node.activation_function_relu()
            elif node.type is NodeType.OUTPUT:   # output
                node.activation_function_sigmoid()
            # input nodes: no activation

        outputs = []
        for node in self.nodes.values():
            if node.type is NodeType.OUTPUT:
                outputs.append(node.value)

        return outputs


    
    

    








'''
def __init__(self, num_inputs, num_outputs):
        self.connections = list()
        self.connection_list = dict()
        self.nodes = list()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.node_values = dict()
        self.fitness = 0

        for i in range(num_inputs):
            self.nodes.append(Node(0.0, True))
        
        for i in range(num_outputs):
            self.nodes.append(Node(0.0, False))
        
        for i in range(num_inputs):
            for j in range(num_outputs):
                self.add_Connection(self.nodes[i], self.nodes[-j - 1], 1, 0)
        
        

    def add_Connection(self, in_Node, out_Node, weight, innovation):
        self.connections.append(Connection(in_Node, out_Node, weight, True, innovation))
        self.connection_list[self.connections[-1]] = (in_Node, out_Node)

    def recursion(self, node):

        if node.type == True:
            return node.value
        else:
            for connection in self.connections:
                if self.connection_list.get(connection)[1] == node:
                    node.value += self.recursion(self.connection_list.get(connection)[0]) * connection.weight
            node.activation_function()
            self.node_values[node] = node.value
            return node.value

    def activate(self):
        outputs = list()
        final_node = Node(0, None)
        outputs = list()
        
        #for node in self.nodes:
        #    if node.type == False:
        #        self.connections.append(Connection(node, final_node, 1, True, -1))
        #        self.connection_list[self.connections[-1]] = (node, final_node)
        
        

        self.recursion(final_node)
        for node in self.nodes:
            if node.type == False:
                outputs.append(node.value)
        return outputs
        
    
    def activate_inputs(self, inputs):
        outputs = list()
        final_node = Node(0, None)

        input_counter = 0
        for node in self.nodes:
            
            #if node.type == False:
            #    self.connections.append(Connection(node, final_node, 1, True, -1))
            #    self.connection_list[self.connections[-1]] = (node, final_node)
            
            if node.type == True:
                node.value = inputs[input_counter]
                input_counter += 1

        self.recursion(final_node)
        for node in self.nodes:
            if node.type == False:
                outputs.append(node.value)
        return outputs

if __name__ == "__main__":
    print(5)

    g = Genome(3, 2)
    g.nodes[0].value = 1
    g.nodes[1].value = 3
    g.nodes[2].value = 3
    print(g.activate())
    print(g.node_values)

    g2 = Genome(3, 2)
    l = [1, 3, 3]
    print(g2.activate_inputs(l))
    print(g2.node_values)
'''

    

        


