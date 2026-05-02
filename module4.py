# module4.py

# Step 1: Read N
N = int(input("Enter a positive integer N: "))

numbers = []

# Step 2: Read N numbers
for i in range(N):
    num = int(input(f"Enter number {i+1}: "))
    numbers.append(num)

# Step 3: Read X
X = int(input("Enter integer X: "))

# Step 4: Find X
if X in numbers:
    print(numbers.index(X) + 1)  # +1 because index starts at 0
else:
    print(-1)
