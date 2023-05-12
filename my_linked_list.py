class Node:
    def __init__(self, doc_id):
        self.next = None
        self.indexes = list()
        self.doc_id: int = doc_id

    def count(self):  # tf
        return len(self.indexes)


# this linked list is sorted based on the doc_id
class LinkedList:
    """
    Attributes
    ----------
    count : int
        total number of indexes inside the linked list which is sum of count from each Node
    
    Methods
    -------
    insert(new_node: Node)
        inserts a node in the correct index in the linked list, that is doc_id incrementally
    get(doc_id: int)
        should return the node with the doc_id attribute equal to doc_id argument
    """

    def __init__(self):
        self.head: Node = None
        self.count = 0
        self.size = 0  # idf

    def insert(self, new_node: Node):
        self.count += new_node.count()
        if self.head is None:
            self.head = new_node
            return
        curr_node = self.head
        while curr_node.next:  # and curr_node.next.doc_id < new_node.doc_id:
            curr_node = curr_node.next
        new_node.next = curr_node.next
        curr_node.next = new_node
        self.size += 1

    def remove(self, node: Node):
        prev = None
        curr = self.head
        while curr and curr is not node:
            prev = curr
            curr = curr.next
        if curr is node:
            if curr is self.head:
                self.head = self.head.next
            else:
                prev.next = curr.next
            self.size -= 1

    def get(self, doc_id: int) -> Node:
        curr = self.head
        while curr and curr.doc_id != doc_id:
            curr = curr.next
        return curr

    def print_list(self):
        curr_node = self.head
        while curr_node:
            print(f"<doc_id: {curr_node.doc_id}, count: {curr_node.count()}, indexes: {curr_node.indexes}>")
            curr_node = curr_node.next