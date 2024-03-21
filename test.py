from random import randint
from jsonargparse import CLI


class Main:
    def __init__(self, max_prize: int = 100):
        """
        Args:
            max_prize: Maximum prize that can be awarded.
        """
        self.max_prize = max_prize

    def person(self, name: str):
        """
        Args:
            name: Name of winner.
        """
        return f"{name} won {randint(0, self.max_prize)}€!"


if __name__ == "__main__":
    print(CLI(Main, as_positional=False))