class Counter:

    def __init__(self):
        self.current_innovation = 0
    
    def get_current_innovation(self):
        i = self.current_innovation
        self.current_innovation += 1
        return i
    
    def subtract_innovation(self):
        self.current_innovation -= 1