import cPickle as pickle

def load_pickle(file_path):
    data = None
    with open (file_path,"rb") as dumpFile:
        data = pickle.load(dumpFile)
    return data

def save_pickle(file_path, data):
    with open (file_path,"wb") as dumpFile:
        pickle.dump(data, dumpFile, pickle.HIGHEST_PROTOCOL)

def count_enum(words):
    wdict = {}
    get = wdict.get
    for word in words:
        wdict[word] = get(word, 0) + 1
    return wdict
