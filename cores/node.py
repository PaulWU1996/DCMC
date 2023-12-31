class Node():
    def __init__(self, id:int, neighbor_l:list, ):
        self.id = id
        self.neighbors = neighbor_l

        self.recieve_buffer = []
        self.send_buffer = []

    def communicate(self):
        # TODO: implement the communication between nodes, share buffer to others
        pass

    def update(self):
        # TODO: implement the update of the node
        pass