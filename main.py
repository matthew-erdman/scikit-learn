from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from time import time


def testModel(classifier, parameters, xMinMax, y):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(100):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time given model
        startTime = time()
        model = classifier(**parameters).fit(trainX, trainY)
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
    print(f"{averageTime:.3f} ms average training time")


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
    logisticParameters = {"max_iter": 1000}
    ridgeParameters = {"max_iter": 1000}
    SGDParameters = {"max_iter": 1000}
    perceptronParameters = {"max_iter": 1000}
    SVCParameters = {"max_iter": 2000}  # 1000 cap can be occasionally hit
    linearSVCParameters = {"max_iter": 1000}

    processData = pd.read_csv("data/setapProcessT9.csv", comment='#')   # read in T9 data from milestones 1-5
    processData = processData.reindex(columns=allowedAttributes)        # drop extra attributes using whitelist
    x = processData[processData.columns[:-1]]                           # extract team process data, exclude final grade
    y = processData[processData.columns[-1]]                            # extract team process grade, ground truth
    xMinMax = MinMaxScaler().fit_transform(x)                           # scale and transform x using minmax

    # test and score models
    allModelPerformance = []
    allModelPerformance.append(("Logistic Regression", *testModel(LogisticRegression, logisticParameters, xMinMax, y)))
    allModelPerformance.append(("Ridge Classifier", *testModel(RidgeClassifier, ridgeParameters, xMinMax, y)))
    allModelPerformance.append(("SGD Classifier", *testModel(SGDClassifier, SGDParameters, xMinMax, y)))
    allModelPerformance.append(("Perceptron", *testModel(Perceptron, perceptronParameters, xMinMax, y)))
    allModelPerformance.append(("Linear SVC", *testModel(LinearSVC, linearSVCParameters, xMinMax, y)))
    for kernel in ["linear", "rbf", "poly"]:
        allModelPerformance.append((f"SVC: {kernel} kernel", *testModel(SVC, SVCParameters, xMinMax, y)))

    # search SVC hyper-parameter space for highest accuracy kernel and parameters
    parameterSpace = [
        {"kernel": ["linear"], "C": [.001, .01, .1, 1, 10, 100, 1000]},
        {"kernel": ["poly"], "C": [.001, .01, .1, 1, 10, 100, 1000], "degree": [2, 3, 4, 5]},
        {"kernel": ["rbf"], "C": [.001, .01, .1, 1, 10, 100, 1000], "gamma": [.01, .1, 1, 10, 100]}
    ]
    gridSearch = GridSearchCV(SVC(), parameterSpace)
    gridResults = gridSearch.fit(xMinMax, y)
    allModelPerformance.append((f"SVC & Grid Search: {gridResults.best_params_}", *testModel(SVC, gridResults.best_params_, xMinMax, y)))

    # sort models by average accuracy descending
    allModelPerformance = sorted(allModelPerformance, key=lambda x: sum(x[1])/len(x[1]), reverse=True)

    # print summary
    print("\n---------------------------------\n\t\t\tSUMMARY\n---------------------------------")
    for model in allModelPerformance:
        printModelSummary(model)



main()

