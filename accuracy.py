def read_actual():
    newDict = {}
    i = 0
    with open('/content/drive/MyDrive/predictions/actual.txt', 'r') as infile:
        for line in infile:
            line = line.strip()
            phrase = line.split()
            newDict[i] = phrase[0]
            i = i+1

    return newDict


def read_predicted():
    newDict = {}
    j = 0
    with open('/content/drive/MyDrive/predictions/predicted.txt', 'r') as infile:
        for line in infile:
            line = line.strip()
            phrase = line.split()
            newDict[j] = phrase[0]
            j = j+1

    return newDict


if __name__ == "__main__":
    actual = read_actual()
    predicted = read_predicted()
    print(actual)
    print(predicted)
    n = len(actual)
    correct = 0
    incorrect = 0
    for i in range(0, n):
        if actual[i] == predicted[i] :
            correct = correct+1
        else:
            incorrect = incorrect+1

    accuracy=(correct/(correct+incorrect))*100
    print("Total Values :",n)
    print("Correct Values :", correct)
    print("Incorrect Values : ", incorrect)
    print("Accuracy of Model : ",accuracy,"%")