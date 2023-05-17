# first line: 17
@mem.cache
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]
