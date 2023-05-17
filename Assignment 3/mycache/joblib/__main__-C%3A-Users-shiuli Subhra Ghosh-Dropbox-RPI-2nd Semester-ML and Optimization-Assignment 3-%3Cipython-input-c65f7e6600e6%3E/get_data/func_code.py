# first line: 1
@mem.cache
def get_data():
    data = load_svmlight_file("a9a.txt")
    return data[0], data[1]
