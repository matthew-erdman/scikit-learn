from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import time


def testLogistic(xMinMax, y):
    # test multiple runs with random test/train splits to average accuracy and times
    allAccuracy = []
    allTime = []
    for i in range(20):
        # reserve 20% of data for testing, split 80% for training
        trainX, testX, trainY, testY = train_test_split(xMinMax, y, test_size=0.2, shuffle=True)

        # train and time logistic regression model
        startTime = time.time()
        logistic = LogisticRegression(max_iter=1000).fit(trainX, trainY)
        endTime = time.time()

        # score logistic regression model
        logisticPrediction = logistic.predict(testX)
        accuracy = accuracy_score(testY, logisticPrediction)

        # record and print logistic regression performance
        allAccuracy.append(accuracy * 100)
        allTime.append(float(endTime - startTime))
        # print(f"Logistic Regression Trial {i + 1} \n---------------------")
        # print(f"{allAccuracy[-1]:.2f}% Accurate")
        # print(f"{allTime[-1]:.3f} Seconds")

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

    processData = pd.read_csv("data/setapProcessT9.csv", comment='#')   # read in T9 data from milestones 1-5
    processData = processData.reindex(columns=allowedAttributes)        # drop extra attributes using whitelist
    x = processData[processData.columns[:-1]]                           # extract team process data, exclude final grade
    y = processData[processData.columns[-1]]                            # extract team process grade, ground truth
    xMinMax = preprocessing.MinMaxScaler().fit_transform(x)             # scale and transform x using minmax

    # test logistic regression model
    logisticAccuracy, logisticTime = testLogistic(xMinMax, y)
    logisticAverageAccuracy = sum(logisticAccuracy) / len(logisticAccuracy)
    logisticAverageTime = sum(logisticTime) / len(logisticTime)

    print(f"Logistic Regression {i + 1} Trial Total \n---------------------")
    print(f"{logisticAverageAccuracy:.2f}% Accurate")
    print(f"{logisticAverageTime:.3f} Seconds")

    # with open("processData.html", "w") as out:
    #     out.write(processData.to_html())


main()

