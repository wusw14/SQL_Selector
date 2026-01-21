from typing import Any


class Memory:
    def __init__(self):
        self.memory = {}

    def add(self, item_name: str, item_value: Any):
        self.memory[item_name] = item_value

    def get(self, item_name):
        return self.memory[item_name]
