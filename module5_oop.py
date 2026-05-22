class NumberStore:
    def __init__(self):
        self._numbers = []

    def add_number(self, number: int) -> None:
        self._numbers.append(number)

    def add_numbers(self, count: int) -> None:
        for _ in range(count):
            value = int(input())
            self.add_number(value)

    def find_first_index(self, target: int) -> int:
        for index, value in enumerate(self._numbers, start=1):
            if value == target:
                return index
        return -1


def main() -> None:
    n = int(input())
    store = NumberStore()
    store.add_numbers(n)
    x = int(input())
    print(store.find_first_index(x))


if __name__ == "__main__":
    main()
