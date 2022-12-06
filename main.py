from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from time import time
from statistics import mean

def readProcessData(filename):
    # attribute whitelist, discards NaN containing columns and some computed variables (standard deviation, other averages, etc.)
    allowedAttributes = [
        "meetingHoursTotal", "averageMeetingHoursTotalByWeek", "averageMeetingHoursTotalByStudent",

        "inPersonMeetingHoursTotal", "averageInPersonMeetingHoursTotalByWeek",
        "averageInPersonMeetingHoursTotalByStudent",

        "nonCodingDeliverablesHoursTotal", "averageNonCodingDeliverablesHoursTotalByWeek",
        "averageNonCodingDeliverablesHoursTotalByStudent",

        "codingDeliverablesHoursTotal", "averageCodingDeliverablesHoursTotalByWeek",
        "averageCodingDeliverablesHoursTotalByStudent",

        "helpHoursTotal", "averageHelpHoursTotalByWeek", "averageHelpHoursTotalByStudent",

        "commitMessageLengthTotal", "averageCommitMessageLengthTotalByWeek",

        "commitCount", "averageCommitCountByWeek", "averageCommitCountByStudent",

        "uniqueCommitMessageCount", "averageUniqueCommitMessageCountByWeek",
        "averageUniqueCommitMessageCountByStudent",

        "commitMessageLengthAverage", "teamMemberCount", "issueCount", "onTimeIssueCount", "lateIssueCount",

        "SE Process grade",
    ]

    # read in data and drop extra attributes using whitelist
    processData = pd.read_csv(filename, comment='#').reindex(columns=allowedAttributes)
    return processData


def printModelSummary(modelPerformance):
    averageAccuracy = mean(modelPerformance[1])
    averageTime = mean(modelPerformance[2]) * 1000
    print(f"{modelPerformance[0]}\n-------------------------------")
    print(f"{averageAccuracy:.1f}% average predictive accuracy")
    print(f"{min(modelPerformance[1]):.1f}% to {max(modelPerformance[1]):.1f}% (range: {max(modelPerformance[1])-min(modelPerformance[1]):.1f}%)")
    print(f"{averageTime:.3f} ms average model training time")


def testModel(classifier, parameters, xMinMax, y):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(10):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.3, shuffle=True)

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


def main():
    # set up model parameters, max iteration counts
    logisticParameters = {"max_iter": 1000}
    ridgeParameters = {"max_iter": 1000}
    SGDParameters = {"max_iter": 1000}
    perceptronParameters = {"max_iter": 1000}
    SVCParameters = {"max_iter": 2000}  # 1000 cap can be occasionally hit
    linearSVCParameters = {"max_iter": 1000}

    processData = readProcessData("data/setapProcessT9.csv")    # read in T9 data from milestones 1-5 as a dataframe
    x = processData[processData.columns[:-1]]                   # extract team process data, exclude final grade
    y = processData[processData.columns[-1]]                    # extract team process grade, ground truth
    xMinMax = MinMaxScaler().fit_transform(x)                  # scale and transform process data using minmax

    # test and score models
    startTime = time()
    allModelPerformance = []
    allModelPerformance.append(("Logistic Regression", *testModel(LogisticRegression, logisticParameters, xMinMax, y)))
    allModelPerformance.append(("Ridge Classifier", *testModel(RidgeClassifier, ridgeParameters, xMinMax, y)))
    allModelPerformance.append(("Perceptron", *testModel(Perceptron, perceptronParameters, xMinMax, y)))
    allModelPerformance.append(("Linear SVC", *testModel(LinearSVC, linearSVCParameters, xMinMax, y)))
    for loss in ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]:
        lossParameters = SGDParameters.copy()
        lossParameters["loss"] = loss
        allModelPerformance.append((f"SGD: {loss} loss", *testModel(SGDClassifier, lossParameters, xMinMax, y)))
    for kernel in ["linear", "rbf", "poly"]:
        kernelParameters = SVCParameters.copy()
        kernelParameters["kernel"] = kernel
        allModelPerformance.append((f"SVC: {kernel} kernel", *testModel(SVC, kernelParameters, xMinMax, y)))

    # search SVC hyper-parameter space for highest accuracy kernel and parameters
    SVCParameterSpace = [
        {"kernel": ["linear"], "C": [.001, .01, .1, 1, 10, 100, 1000]},
        {"kernel": ["poly"], "C": [.001, .01, .1, 1, 10, 100, 1000], "degree": [2, 3, 4]},
        {"kernel": ["rbf"], "C": [.001, .01, .1, 1, 10, 100, 1000], "gamma": [.01, .1, 1, 10, 100]},
    ]
    SVCGridSearch = GridSearchCV(SVC(), SVCParameterSpace)
    SVCGridResults = SVCGridSearch.fit(xMinMax, y)
    allModelPerformance.append((f"SVC & Grid Search: {SVCGridResults.best_params_}", *testModel(SVC, SVCGridResults.best_params_, xMinMax, y)))

    # search for highest accuracy SGD loss function
    SGDParameterSpace = {
        "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
        "penalty": ["elasticnet"],
        "l1_ratio": [0, .2, .4, .6, .8, 1],
    }
    SGDGridSearch = GridSearchCV(SGDClassifier(), SGDParameterSpace)
    SGDGridResults = SGDGridSearch.fit(xMinMax, y)
    allModelPerformance.append((f"SGD & Grid Search: {SGDGridResults.best_params_}", *testModel(SGDClassifier, SGDGridResults.best_params_, xMinMax, y)))
    endTime = time()

    # sort models by average accuracy descending
    allModelPerformance = sorted(allModelPerformance, key=lambda x: mean(x[1]), reverse=True)

    # print summary
    print("\n---------------------------------\n\t\t\tSUMMARY\n---------------------------------")
    print(f"Dataset of {processData.shape[0]} records by {processData.shape[1]} variables")
    print(f"Tested {len(allModelPerformance)} classification methods with {len(allModelPerformance[0][1])} random splits each")
    print(f"Total time to train and test all models: {(endTime - startTime) * 1000:.1f} ms")
    print("\nPress enter to start stepping through the classifiers (ranked most accurate on average to least accurate)")
    for i in range(len(allModelPerformance)):
        input("...")
        print(f"\n{i + 1} of {len(allModelPerformance)}")
        printModelSummary(allModelPerformance[i])



main()