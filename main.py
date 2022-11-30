from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import time


def testLogistic(xMinMax, y, maxIterations):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(100):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time logistic regression model
        startTime = time.time()
        model = LogisticRegression(max_iter=maxIterations).fit(trainX, trainY)
        endTime = time.time()

        # score logistic regression model
        prediction = model.predict(testX)
        accuracy = accuracy_score(testY, prediction)

        # record and print logistic regression performance
        allAccuracy.append(accuracy * 100)
        allTime.append(float(endTime - startTime))
        print("---------------------")
        print(f"Logistic Regression Trial {i + 1}")
        print(f"Accuracy: {allAccuracy[-1]:.2f}%")
        print(f"Seconds: {allTime[-1]:.3f}")

    return allAccuracy, allTime


def testRidge(xMinMax, y, maxIterations):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(100):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time ridge classifier
        startTime = time.time()
        model = RidgeClassifier(max_iter=maxIterations).fit(trainX, trainY)
        endTime = time.time()

        # score ridge classifier
        prediction = model.predict(testX)
        accuracy = accuracy_score(testY, prediction)

        # record and print ridge classifier performance
        allAccuracy.append(accuracy * 100)
        allTime.append(float(endTime - startTime))
        print("---------------------")
        print(f"Ridge Classifier Trial {i + 1}")
        print(f"Accuracy: {allAccuracy[-1]:.2f}%")
        print(f"Seconds: {allTime[-1]:.3f}")

    return allAccuracy, allTime


def testSGD(xMinMax, y, maxIterations):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(100):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time sgd classifier
        startTime = time.time()
        model = SGDClassifier(max_iter=maxIterations).fit(trainX, trainY)
        endTime = time.time()

        # score sgd classifier
        prediction = model.predict(testX)
        accuracy = accuracy_score(testY, prediction)

        # record and print sgd classifier performance
        allAccuracy.append(accuracy * 100)
        allTime.append(float(endTime - startTime))
        print("---------------------")
        print(f"SGD Classifier Trial {i + 1}")
        print(f"Accuracy: {allAccuracy[-1]:.2f}%")
        print(f"Seconds: {allTime[-1]:.3f}")

    return allAccuracy, allTime


def testPerceptron(xMinMax, y, maxIterations):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(100):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time perceptron
        startTime = time.time()
        model = Perceptron(max_iter=maxIterations).fit(trainX, trainY)
        endTime = time.time()

        # score perceptron
        prediction = model.predict(testX)
        accuracy = accuracy_score(testY, prediction)

        # record and print perceptron performance
        allAccuracy.append(accuracy * 100)
        allTime.append(float(endTime - startTime))
        print("---------------------")
        print(f"Perceptron Trial {i + 1}")
        print(f"Accuracy: {allAccuracy[-1]:.2f}%")
        print(f"Seconds: {allTime[-1]:.3f}")

    return allAccuracy, allTime


def main():
    # attribute whitelist, discards some computed variables (standard deviation, other averages, etc.)
    allowedAttributes = ["teamMemberCount", "meetingHoursTotal", "inPersonMeetingHoursTotal", "nonCodingDeliverablesHoursTotal",
                         "codingDeliverablesHoursTotal", "helpHoursTotal", "averageMeetingHoursTotalByWeek",
                         "averageInPersonMeetingHoursTotalByWeek", "averageNonCodingDeliverablesHoursTotalByWeek",
                         "averageCodingDeliverablesHoursTotalByWeek", "averageHelpHoursTotalByWeek", "averageMeetingHoursTotalByStudent",
                         "averageInPersonMeetingHoursTotalByStudent", "averageNonCodingDeliverablesHoursTotalByStudent",
                         "averageCodingDeliverablesHoursTotalByStudent", "averageHelpHoursTotalByStudent", "commitCount",
                         "uniqueCommitMessageCount", "commitMessageLengthAverage", "averageCommitCountByWeek",
                         "averageUniqueCommitMessageCountByWeek", "averageCommitCountByStudent", "averageUniqueCommitMessageCountByStudent",
                         "issueCount", "onTimeIssueCount", "lateIssueCount", "SE Process grade"]

    # set max iteration counts
    logisticIterations = 1000
    ridgeIterations = 1000
    SGDIterations = 1000
    perceptronIterations = 1000

    processData = pd.read_csv("data/setapProcessT9.csv", comment='#')   # read in T9 data from milestones 1-5
    processData = processData.reindex(columns=allowedAttributes)        # drop extra attributes using whitelist
    x = processData[processData.columns[:-1]]                           # extract team process data, exclude final grade
    y = processData[processData.columns[-1]]                            # extract team process grade, ground truth
    xMinMax = preprocessing.MinMaxScaler().fit_transform(x)             # scale and transform x using minmax

    # test logistic regression model
    logisticAccuracy, logisticTime = testLogistic(xMinMax, y, logisticIterations)
    logisticAverageAccuracy = sum(logisticAccuracy) / len(logisticAccuracy)
    logisticAverageTime = (sum(logisticTime) / len(logisticTime)) * 1000

    # test ridge classifier
    ridgeAccuracy, ridgeTime = testRidge(xMinMax, y, ridgeIterations)
    ridgeAverageAccuracy = sum(ridgeAccuracy) / len(ridgeAccuracy)
    ridgeAverageTime = (sum(ridgeTime) / len(ridgeTime)) * 1000

    # test sgd classifier
    SGDAccuracy, SGDTime = testSGD(xMinMax, y, SGDIterations)
    SGDAverageAccuracy = sum(SGDAccuracy) / len(SGDAccuracy)
    SGDAverageTime = (sum(SGDTime) / len(SGDTime)) * 1000

    # test perceptron
    perceptronAccuracy, perceptronTime = testPerceptron(xMinMax, y, perceptronIterations)
    perceptronAverageAccuracy = sum(perceptronAccuracy) / len(perceptronAccuracy)
    perceptronAverageTime = (sum(perceptronTime) / len(perceptronTime)) * 1000

    # print averages
    print("\n---------------------------------\n\t\t\tAVERAGES\n---------------------------------")
    print(f"\nLogistic Regression \n---------------------")
    print(f"{logisticAverageAccuracy:.1f}% predictive accuracy ({len(logisticAccuracy)} trials)")
    print(f"{logisticAverageTime:.3f} ms to train")
    print(f"{logisticIterations} max iterations")

    print(f"\nRidge Classifier \n---------------------")
    print(f"{ridgeAverageAccuracy:.1f}% predictive accuracy ({len(ridgeAccuracy)} trials)")
    print(f"{ridgeAverageTime:.3f} ms to train")
    print(f"{ridgeIterations} max iterations")

    print(f"\nSGD Classifier \n---------------------")
    print(f"{SGDAverageAccuracy:.1f}% predictive accuracy ({len(SGDAccuracy)} trials)")
    print(f"{SGDAverageTime:.3f} ms to train")
    print(f"{SGDIterations} max iterations")

    print(f"\nPerceptron \n---------------------")
    print(f"{perceptronAverageAccuracy:.2f}% predictive accuracy ({len(perceptronAccuracy)} trials)")
    print(f"{perceptronAverageTime:.3f} ms to train")
    print(f"{perceptronIterations} max iterations")

    # with open("processData.html", "w") as out:
    #     out.write(processData.to_html())


main()

