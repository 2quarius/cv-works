from PyQt5.QtGui import QImage

class Node(object):
    def __init__(self,data,next = None):
        self.data = data
        self.next = next
class Stack(object):
    def __init__(self,top = None):
        self.top = top
    def push(self,data):
        self.top = Node(data,self.top)
    def pop(self):
        if self.top is None:
            return None
        data = self.top.data
        self.top = self.top.next
        return data
    def peek(self):
        return self.top.data if self.top is not None else None
    def isEmpty(self):
        return self.peek() is None
# if __name__=="__main__":
#     stack = Stack()
#     img = QImage()
#     stack.push(img)
#     print(stack.peek())
#     stack.pop()
#     print(stack.isEmpty())
#     print(stack.pop())
