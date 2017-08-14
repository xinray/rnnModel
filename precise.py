import sys

all = 0
right = 0
wrong = 0
for line in open(sys.argv[1]):
    line = line.strip()

    sep = line.split("\t")

    label = sep[1]
    pred = sep[3]

    all += 1

    if label != pred:
        print line
        wrong += 1
    else:
        right += 1

print right, wrong ,all
print right*1.0/all
