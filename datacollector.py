import dataset
xsize = 20
start = 0 # Probably poinless
end = len(dataset.dataset) - (xsize + 2) # Starts at 0 and we need a y value

x_train = [] # 2d
y_train = [] # 1d
for i in range(start, end):
    x = dataset.dataset[i:i+xsize]
    y = dataset.dataset[i+xsize]
    x_train.append(x)
    y_train.append(y)

# Yeah ik this is a bit weird but wtv
def getx():
    return x_train

def gety():
    return y_train
