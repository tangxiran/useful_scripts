
def readTxtExacrtLoss(filename="loss.txt",lossbeginStr="loss is:"
                      ,lossendStr=")"):
    data = []
    with open(filename, "r+") as read:
        data = read.readlines()
    print(data)
    datatorecord = []
    for line in data:
        if line.find(lossbeginStr) > 0:
            res = line.split(lossbeginStr)
            res = res[-1]
            res = res.split(lossendStr)
            res=res[0]
            res = float(res)
            datatorecord.append(res)
    print(datatorecord)
    data = copy.deepcopy(datatorecord)
    data = np.reshape(data, (len(data), 1))
    savenpyasexcel(data, filename.replace(".txt","Change2Excel.xlsx"))
