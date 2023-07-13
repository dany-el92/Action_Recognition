import json

temp = json.load(open('classes.json'))

data = []
for i in temp:
    data.append(set(temp[i]))

for k in range(len(data)):
    for k1 in range(len(data)):
        for k2 in range(len(data)):
            if k1 != k2:
                newVal = set.intersection(data[k1], data[k2])
                #print(k, data[k1], data[k2], "=>", newVal, len(newVal))
                if len(newVal) > 0:
                    data.append(newVal)
                    data[k1] = data[k1] - newVal
                    data[k2] = data[k2] - newVal

resultsTemp = []
for i in data:
    if len(i) > 0:
        list_to_appent = list(i)
        list_to_appent.sort()
        resultsTemp.append(list_to_appent)

# take second element for sort
def takeFirst(elem):
    return elem[0]

resultsTemp.sort(key=takeFirst)

result = {}
i = 1
for val in resultsTemp:
    result[i] = val
    i += 1

with open('groups.json', 'w') as outfile:
    json.dump(result, outfile)