class MethodNotExecutedError(Exception):
    def __init__(self, method_name):
        self.method_name = method_name

    def __str__(self):
        return f"Error: '{self.method_name}' method has not been executed yet."
