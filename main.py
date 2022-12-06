from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from time import time


def testModel(classifier, maxIterations, xMinMax, y, kernel=None):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(100):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time given model
        if kernel:
            startTime = time()
            model = classifier(max_iter=maxIterations, kernel=kernel).fit(trainX, trainY)
            endTime = time()
        else:
            startTime = time()
            model = classifier(max_iter=maxIterations).fit(trainX, trainY)
            endTime = time()

        # measure model performance
        prediction = model.predict(testX)
        accuracy = accuracy_score(testY, prediction)

        # calculate and print performance
        allAccuracy.append(accuracy * 100)
        allTime.append(float(endTime - startTime))
        print("---------------------")
        print(f"{model} Trial {i + 1}")
        print(f"Accuracy: {allAccuracy[-1]:.2f}%")
        print(f"Seconds: {allTime[-1]:.3f}")

    return allAccuracy, allTime


def printModelSummary(model):
    averageAccuracy = sum(model[1]) / len(model[1])
    averageTime = (sum(model[2]) / len(model[2])) * 1000

    print(f"\n{model[0]}\n---------------------")
    print(f"{averageAccuracy:.1f}% average predictive accuracy (range: {min(model[1]):.2f}% to {max(model[1]):.2f}%)")
    print(f"{averageTime:.3f} ms average to train")


# def findBestSVC(xMinMax, y, maxIterations, kernel):
    # # search hyper-parameter space for best cross validation score
    # parameters = [
    #     {"kernel": ["linear"], "C": [.001, .01, .1, 1, 10, 100, 1000]},
    #     {"kernel": ["poly"], "C": [.001, .01, .1, 1, 10, 100, 1000], "degree": [2, 3, 4, 5]},
    #     {"kernel": ["rbf"], "C": [.001, .01, .1, 1, 10, 100, 1000], "gamma": [[.01, .1, 1, 10, 100]]},
    # ]
    # classifier = GridSearchCV(SVC(), parameters)
    # classifier.fit(xMinMax, y)


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
    SVCIterations = 10000  # 1000 cap can be occasionally hit
    linearSVCIterations = 1000

    processData = pd.read_csv("data/setapProcessT9.csv", comment='#')   # read in T9 data from milestones 1-5
    processData = processData.reindex(columns=allowedAttributes)        # drop extra attributes using whitelist
    x = processData[processData.columns[:-1]]                           # extract team process data, exclude final grade
    y = processData[processData.columns[-1]]                            # extract team process grade, ground truth
    xMinMax = MinMaxScaler().fit_transform(x)                           # scale and transform x using minmax

    # test and score models
    allModelPerformance = []
    allModelPerformance.append(("Logistic Regression", *testModel(LogisticRegression, logisticIterations, xMinMax, y)))
    allModelPerformance.append(("Ridge Classifier", *testModel(RidgeClassifier, ridgeIterations, xMinMax, y)))
    allModelPerformance.append(("SGD Classifier", *testModel(SGDClassifier, SGDIterations, xMinMax, y)))
    allModelPerformance.append(("Perceptron", *testModel(Perceptron, perceptronIterations, xMinMax, y)))
    allModelPerformance.append(("Linear SVC", *testModel(LinearSVC, linearSVCIterations, xMinMax, y)))
    for kernel in ["linear", "rbf", "poly"]:
        allModelPerformance.append((f"SVC: {kernel} kernel", *testModel(SVC, SVCIterations, xMinMax, y, kernel=kernel)))

    # print summary
    print("\n---------------------------------\n\t\t\tSUMMARY\n---------------------------------")
    for model in allModelPerformance:
        printModelSummary(model)



main()

