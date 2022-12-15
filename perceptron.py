import numpy as np
np.warnings.filterwarnings('ignore')
import random

""" Function that processes the data. It takes in the arguments dataFilename and classLabel.
dataFilename is the path of the data file. classLabel is a vector that consists of 3 values, 
values that can be -1,+1 or 0. The dataProcessing function gives the first value of classLabel to
data occurence with class class-1, the second value of classLabel to data occurence with class-2 
and the third value of classLabel to the data occurence labeled with class-3. The function returns 
the processed data."""
def dataProcessing(dataFilename,classLabel):
    #import the training and test data
    
    dataNpX = np.genfromtxt(dataFilename, delimiter=',', skip_header = 1, usecols = (0,1,2,3),dtype=float) #x values
    dataNpC = np.genfromtxt(dataFilename, delimiter=',', skip_header = 1, usecols = (4),dtype=str) #class values

    newDataNpC = dataNpC

    #change the classes in daraNpC to the classLabel(+1,-1,0)
    for i in range(0,len(dataNpC)):
        if dataNpC[i] == "class-1":
            newDataNpC[i] = classLabel[0]
        elif dataNpC[i] == "class-2":
            newDataNpC[i] = classLabel[1]
        elif dataNpC[i] == "class-3":
            newDataNpC[i] = classLabel[2]
    
    #combine dataNpX & dataNpC
    dataReady = np.c_[dataNpX,dataNpC]
    #cast to float
    d = dataReady.astype(float)
    #randomize x-values in data
    np.random.shuffle(d)
    #return the data
    return d

""" Function that uses the perceptron algorithm to train data. It also calculates the training accuracy
by counting correct and incorrect occurences of the activation score. The inputs of the functions are
maxIter that is how many iterations the training will make and trainData that is the processed training data.
The outputs are b and w, respectively the bias and the weigth vector."""
def perceptronTrain(maxIter,trainData):
    D = len(trainData)
    l = len(trainData[0])-1
    w = np.zeros((l), dtype=float)
    b = 0.0 
    notcorrect = 0
    correct = 0
    for i in range(0,maxIter):
        for x in trainData:
            y = x[4]
            if y == 0:
                continue
            wt = np.transpose(w)
            x1 = x[:4] #slice the data array so we only have the x values
            wtx = np.dot(wt,x1)
            a = wtx+b
            ya = y*a
            if ya <= 0:
                notcorrect +=1
                for r in range(0,l):
                    w[r] = w[r] + y*x[r]
                b = b+y
            else:
                correct +=1
    accuracy = (correct/(correct+notcorrect))*100
    print("Training accuracy:",accuracy,"%.")
    return b,w
    
""" Same as the perceptronTrain function with an additional input, lambdaArray. The update rule in the training
algorithm is also different. LamdaArray is an array
containing the lambda values we want to look at and choose a lambda value from. First the function splits
the training data into two parts, that are split roughly 80%/20% and we call the data trainData80 and valData20.
For each lamda we train the trainData80 20 times (maxIter), and test the data on valData20. When this 
has been done for all the lambda values in lambdaArray, the lamda which produced the highest accuracy on
testing the valData20 is chosen as our lambda (chosenLamda). Then we train the whole training data with
our chosen lambda and return the b and w values for that. """
def perceptronTrainL2(maxIter,trainData,lambdaArray):
    D = len(trainData)
    #split the train data to training(80%) and validation(20%).
    #Let 96 rows of the data be in trainData80 and 24 rows be in valData20.
    trainData80 = trainData[:96]
    valData20 = trainData[97:]

    l = len(trainData80[0])-1
    w = np.zeros((l), dtype=float)
    b = 0.0 
    counter=0
    accuracyArr = np.zeros(len(lambdaArray), dtype=float) #array that keeps record of the accuracies with different lamba values
    for o in lambdaArray:
        notcorrect = 0
        correct = 0
        for i in range(0,maxIter):#here we train the data with the train80 data
            for x in trainData80:
                y = x[4]
                if y == 0:
                    continue
                wt = np.transpose(w)
                x1 = x[:4] #slice the data array so we only have the x values
                wtx = np.dot(wt,x1)
                a = wtx+b
                ya = y*a
                if ya <= 0:
                    for r in range(0,l):
                        w[r] = (1-(2*o))*w[r] + y*x[r] #given that mu = 1
                    b = b+y
        #here we test the validation data and keep track of accuracies:
        x = np.delete(valData20, 4, 1) #create array x that is the test data without the labels
        wrong = 0
        right = 0
        c=0
        for i in x:
            wtx = np.dot(wt,i)
            #compute a, if a is negative then we predicted wrong
            a = wtx + b
            if a > 0:
                prediction= 1.0
            else:
                prediction=-1.0
            if prediction != valData20[c][4]:  
                wrong +=1
            else:
                right +=1
            c+=1
        accuracy = (right/(right+wrong))*100
        accuracyArr[counter]=accuracy
        counter+=1
    laMax = np.argmax(accuracyArr)
    
    #print("accuracy array:",accuracyArr)
    #print("Lambda:",lambdaArray[laMax])
    #Now we do it all again but we do as before, using the laMax lambda we got from earlier
    D = len(trainData)
    l = len(trainData[0])-1
    w = np.zeros((l), dtype=float)
    b = 0.0 
    notcorrect = 0
    correct = 0
    chosenLambda = lambdaArray[laMax]
    for i in range(0,maxIter):
        for x in trainData:
            y = x[4]
            if y == 0:
                continue
            wt = np.transpose(w)
            x1 = x[:4] #slice the data array so we only have the x values
            wtx = np.dot(wt,x1)
            a = wtx+b
            ya = y*a
            if ya <= 0:
                notcorrect +=1
                for r in range(0,l):
                    w[r] = (1-(2*chosenLambda)*w[r]) + y*x[r] #given that mu = 1
                b = b+y
            else:
                correct +=1
    accuracy = (correct/(correct+notcorrect))*100
    print("Training accuracy:",accuracy,"%.")
    return b,w

""" Function that takes in inputs b,w and testData. b is bias, w is the weight vector and testData is our 
processed test data. In the end we calculate the accuracy of the testing."""
def perceptronTest(b,w,testData):
    wt = w #we use the weight vector created in the perceptronTrain function
    x = np.delete(testData, 4, 1) #create array x that is the test data without the labels
    wrong = 0
    right = 0
    c=0
    for i in x:
        wtx = np.dot(wt,i)
        #compute a, if a is negative then we predicted wrong
        a = wtx + b
        if a > 0:
            prediction= 1.0
        else:
            prediction=-1.0
        if prediction != testData[c][4]:  
            wrong +=1
        else:
            right +=1
        c+=1
    accuracy = (right/(right+wrong))*100
    print("Testing accuracy:",accuracy,"%")

""" Function that takes in inputs bA,wA and testData. bA is a vector that contains the bias values for all 
the three different 1vsRest cases, wA is an array containing the weight vectors for the different cases and 
testData is our processed test data. In the end we calculate the accuracy of the testing."""
def perceptronTest1vR(bA,wA,testData):
    #wt = wA #we use the weight vector created in the perceptronTrain function
    x = np.delete(testData, 4, 1) #create array x that is the test data without the labels
    wrong = 0
    right = 0
    c=0
    aArr = np.zeros(shape=(4))
    for i in x:
        s=0
        for w in wA: #this is to loop over all the weights and store all the a values
            wtx = np.dot(w,i)
            #compute a, if a is negative then we predicted wrong
            #print("bA[s]",bA[s])
            a = wtx + bA[s]
            aArr[s]=a
            s+=1
        a = np.amax(aArr) #we choose a as the maximum value we got
        prediction = 0
        if aArr[0] == a:#class1
            prediction = 1
        elif aArr[1] == a:#class2
            prediction = 2
        elif aArr[2] == a:#class3
            prediction = 3
        if prediction == testData[c][4]:
            right +=1
        else:
            wrong +=1
        c+=1
    accuracy = (right/(right+wrong))*100
    print("Testing accuracy:",accuracy,"%")

trainDataPath = 'train.data'
testDataPath = 'test.data'

print('---------------------------------------')

classLabelArr = np.array([[1.0,-1.0,0.0],[0.0,1.0,-1.0],[1.0,0.0,-1.0],[1.0,-1.0,-1.0],[-1.0,1.0,-1.0],[-1.0,-1.0,1.0]])
classLabelCase = np.array([["class 1 and class 2:"],["class 2 and class 3:"],["class 1 and class 3:"],["class 1 vs rest:"],["class 2 vs rest:"],["class 3 vs rest:"]])

print("Question 3:")
for p in range(0,3):
    print(classLabelCase[p])
    classLabel3 = classLabelArr[p]
    trainData3 = dataProcessing(trainDataPath,classLabel3)
    testData3 = dataProcessing(testDataPath,classLabel3)
    resultTrain3 = perceptronTrain(20,trainData3)
    b3=resultTrain3[0]
    w3=resultTrain3[1]
    resultTest3 = perceptronTest(b3,w3,testData3)
print('---------------------------------------')
print("Question 4:")
wArr = np.zeros(shape=(3,4)) #array that keeps track of the weight vectors from training in 1vsrest. 3 different weight vectors, with 4 values in each
bArr = np.zeros(shape=(3)) #array that keeps track of the biases from training in 1vsrest. We end up with 3 bias values.
for p in range(3,6):#do this 3 times, first for 1vsrest, then for 2vsrest, then 3vsrest
    print(classLabelCase[p])
    classLabel3 = classLabelArr[p] #first the label from 1vsrest, then 2vsrest then 3vsrest
    trainData3 = dataProcessing(trainDataPath,classLabel3) #process the data with the corresponding labels
    testData3 = dataProcessing(testDataPath,classLabel3) #process the data with the corresponding labels
    resultTrain3 = perceptronTrain(20,trainData3) #process the data with the corresponding labels
    b3=resultTrain3[0]
    bArr[p-3]=b3
    w3=resultTrain3[1]
    wArr[p-3]=w3
testData3 = dataProcessing(testDataPath,[1,2,3]) 
resultTest3 = perceptronTest1vR(bArr,wArr,testData3)
print('---------------------------------------')
print("Question 5:")
lambdaArr = np.array([0.01,0.1,1.0,10.0,100.0]) #array that holds the lambda values that we want to look at
wArr = np.zeros(shape=(3,4)) #array that keeps track of the weight vectors from training in 1vsrest. 3 different weight vectors, with 4 values in each
bArr = np.zeros(shape=(3)) #array that keeps track of the biases from training in 1vsrest. We end up with 3 bias values.
for p in range(3,6):
    print(classLabelCase[p])
    classLabel3 = classLabelArr[p]
    trainData3 = dataProcessing(trainDataPath,classLabel3)
    testData3 = dataProcessing(testDataPath,classLabel3)
    resultTrain3 = perceptronTrainL2(20,trainData3,lambdaArr)
    b3=resultTrain3[0]
    bArr[p-3]=b3
    w3=resultTrain3[1]
    wArr[p-3]=w3 
testData3 = dataProcessing(testDataPath,[1,2,3]) 
resultTest3 = perceptronTest1vR(bArr,wArr,testData3)