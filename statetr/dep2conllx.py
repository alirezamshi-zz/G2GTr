import os
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))



if len(sys.argv) < 3:
    sys.stderr.write("usage: python substituteConllxPOS.py conllx_file pos_file\n")
    sys.exit(1)

class CoNLLReader:
    def __init__(self, file):
        self.file = file
    def __iter__(self):
        return self

    def __next__(self):
        sent = self.readsent()
        if sent == []:
            raise StopIteration()
        else:
            return sent

    def readsent(self):
        """
        Assuming CoNLL-U format, where the columns are:
        ID FORM LEMMA UPOSTAG XPOSTAG FEATS HEAD DEPREL DEPS MISC
        """
        sent = []
        row_str = self.file.readline().strip()
        while row_str != "":
            row = {}
            columns = row_str.split()
            row["ID"] = int(columns[0])
            row["FORM"] = columns[1]
            row["LEMMA"] = columns[2] if len(columns) > 2 else "_"
            row["UPOSTAG"] = columns[3] if len(columns) > 3 else "_"
            row["XPOSTAG"] = columns[4] if len(columns) > 4 else "_"
            row["FEATS"] = columns[5] if len(columns) > 5 else "_"
            row["HEAD"] = columns[6] if len(columns) > 6 else "_"
            row["DEPREL"] = columns[7] if len(columns) > 7 else "_"
            row["DEPS"] = columns[8] if len(columns) > 8 else "_"
            row["MISC"] = columns[9] if len(columns) > 9 else "_"
            sent.append(row)
            row_str = self.file.readline().strip()
        return sent

    def close(self):
        self.file.close()
        
def write_row(row):
    sys.stdout.write(str(row["ID"]) + "\t")
    sys.stdout.write(row["FORM"] + "\t")
    sys.stdout.write(row["LEMMA"] + "\t")
    sys.stdout.write(row["UPOSTAG"] + "\t")
    sys.stdout.write(row["XPOSTAG"] + "\t")
    sys.stdout.write(row["FEATS"] + "\t")
    sys.stdout.write(str(row["HEAD"]) + "\t")
    sys.stdout.write(row["DEPREL"] + "\t")
    sys.stdout.write(row["DEPS"] + "\t")
    sys.stdout.write(row["MISC"] + "\n")


conll_reader = CoNLLReader(open(sys.argv[1]))
#pos_reader = sys.argv[2])
with open(str(sys.argv[2]+"/dependency/"+sys.argv[3])+'.pkl', 'rb') as f:
    dependencies_total_old = pickle.load(f)
with open(str(sys.argv[2])+"/vocab/"+sys.argv[3]+'.pkl', 'rb') as f:
    parser = pickle.load(f)
#print(dependencies_total[0])
conll_read = []
for conll_sent in conll_reader:
    conll_read.append(conll_sent)

lengths = [len(x)+1 for x in conll_read]

index_sorted = sorted(range(len(lengths)), key=lambda k: lengths[k],reverse=True)
index_new = sorted(range(len(index_sorted)), key=lambda k: index_sorted[k])

dependencies_total = [dependencies_total_old[i] for i in index_new]

assert len(dependencies_total) == len(conll_read)

for conll_sent, dependencies in zip(conll_read, dependencies_total):
    heads = []
    deprels = []
    dependencies.sort(key=lambda row: row[1])
    for item in dependencies:
        heads.append(item[0])
        #print(item)
        if item[2] < 0:
            x = item[2]+parser.n_deprel
        elif item[2] >= parser.n_deprel:
            x = item[2] - parser.n_deprel
        else:
            x = item[2]
        assert 0<= x < parser.n_deprel
        deprels.append(parser.id2tok[x].split(":")[1])
    assert len(heads) == len(deprels)
    if len(conll_sent) > len(heads):
        dif = len(conll_sent) - len(heads)
        heads = heads + [0]*dif
        deprels = deprels + ['-']*dif
    assert len(conll_sent) == len(heads) == len(deprels), "conll:{},pred:{},deprel:{}".format(len(conll_sent),len(heads),len(deprels))
    for conll_row, head, deprel in zip(conll_sent, heads, deprels):
        conll_row["HEAD"] = head
        conll_row["DEPREL"] = deprel
        write_row(conll_row)
    sys.stdout.write("\n")
    
    
conll_reader.close()
