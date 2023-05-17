# first line: 1
@mem.cache
# function to load datasets
def get_data(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]
