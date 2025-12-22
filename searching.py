data = [10, 20, 30, 40, 50]
def linear_search(data, target):
    for i in range(len(data)):
        if data[i] == target:
            return i
    return -1
print("element is found at index : ",linear_search(data,20))