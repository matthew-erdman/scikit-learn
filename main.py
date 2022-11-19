from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    allowedAttributes = ["teamMemberCount", "meetingHoursTotal", "inPersonMeetingHoursTotal", "nonCodingDeliverablesHoursTotal",
                         "codingDeliverablesHoursTotal", "helpHoursTotal", "averageMeetingHoursTotalByWeek",
                         "averageInPersonMeetingHoursTotalByWeek", "averageNonCodingDeliverablesHoursTotalByWeek",
                         "averageCodingDeliverablesHoursTotalByWeek", "averageHelpHoursTotalByWeek", "averageMeetingHoursTotalByStudent",
                         "averageInPersonMeetingHoursTotalByStudent", "averageNonCodingDeliverablesHoursTotalByStudent",
                         "averageCodingDeliverablesHoursTotalByStudent", "averageHelpHoursTotalByStudent", "commitCount",
                         "uniqueCommitMessageCount", "commitMessageLengthAverage", "averageCommitCountByWeek",
                         "averageUniqueCommitMessageCountByWeek", "averageCommitCountByStudent", "averageUniqueCommitMessageCountByStudent",
                         "issueCount", "onTimeIssueCount", "lateIssueCount", "SE Process grade"]

    processData = pd.read_csv("data/setapProcessT9.csv", comment='#')  # read in data from milestones 1-5
    processData = processData.reindex(columns=allowedAttributes)  # drop extra attributes using whitelist

    with open("processData.html", "w") as out:
        out.write(processData.to_html())


main()
