import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


df=pd.read_csv("employee_attrition_dataset.csv",index_col="Employee_ID")
Y=df["Attrition"]
X=df.drop("Attrition",axis=1)





print(df['Gender'].value_counts())
print("-----------------------------------------------------------")
print(df['Marital_Status'].value_counts())
print("-----------------------------------------------------------")
print(df['Department'].value_counts())
print("-----------------------------------------------------------")
print(df['Job_Role'].value_counts())



print(X)
print(df.shape)
print(Y)


print(df.describe())


#Generating standardized data
Numeric_X= X.select_dtypes(include=np.number)
scaler= StandardScaler()
scaler.fit(Numeric_X)
STD = scaler.transform(Numeric_X)
print(np.asarray(STD))




#DATA SPLITTING
X_train, X_test, Y_train, Y_test = train_test_split(STD,Y, test_size=0.2, stratify=Y,random_state=2)
print("X.shape:",X.shape,"X_train.shape:",X_train.shape,"X_test.shape:",X_test.shape)


#Training the Model
classifier = svm.SVC(
    kernel='linear',
    class_weight='balanced'
)

#TRAINING THE SUPPORT VECTOR MACHINE CLASSIFIER
classifier.fit(X_train, Y_train)

# MODEL EVALUATION

#ACCURACY SCORE

#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data:", training_data_accuracy*100,"%")


#accuracy score of the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of the test data:", test_data_accuracy*100,"%")

#Making a Predictive System
Age= int(input("Age:"))
Job_level=int(input("Job_level:"))
Monthly_Income=int(input("Monthly_Income:"))
Hourly_Rate = int(input("Hourly_Rate:"))
Years_at_Company= int(input("Years_At_Company:"))
Years_in_Current_Role= int(input("Years_In_Current_role:"))
Years_Since_Last_Promotion = int(input("Years_Since_Last_Promotion:"))
Work_Life_Balance= int(input("Work_Life_Balance:"))
Job_Satisfaction = int(input("Job_Satisfaction:"))
Performance_Rating = int(input("Performance_Rating:"))
Training_Hours_Last_Year = float(input("Training_Hours_Last_year:"))
Project_Count = int(input("Project_Count:"))
Average_Hours_Worked_Per_Week = int(input("Average_Hours_Worked_Per_Week :"))
Absenteeism= int(input("Absenteeism:"))
Work_Environment_Satisfaction = int(input("Work_Environment_Satisfaction:"))
Relationship_With_Manager = int(input("Relationship_with_Manager:"))
Job_Involvement = int(input("Job Involvement:"))
Distance_From_Home = int(input("Distance_From_Home:"))
Number_of_Companies_Worked = int(input("Number_of_Companies_Worked:"))

input_data = (
    Age,
    Job_level,
    Monthly_Income,
    Hourly_Rate,
    Years_at_Company,
    Years_in_Current_Role,
    Years_Since_Last_Promotion,
    Work_Life_Balance,
    Job_Satisfaction,
    Performance_Rating,
    Training_Hours_Last_Year,
    Project_Count,
    Average_Hours_Worked_Per_Week,
    Absenteeism,
    Work_Environment_Satisfaction,
    Relationship_With_Manager,
    Job_Involvement,
    Distance_From_Home,
    Number_of_Companies_Worked
)

#changing the input_data_to_numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



#Standardize the input Data
STD_DATA=scaler.transform(input_data_reshaped)
print(STD_DATA)

#Prediction
prediction = classifier.predict(STD_DATA)
print(prediction)
print(np.unique(prediction))