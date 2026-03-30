# Adam 

import pandas as pd
import train as t
import predict as p

# confirm sizes of splits
# print(X_train.shape)
# print(X_test.shape)

# START MENU

 
choice = 0

print("SCAM OR LEGIT ?")
print("1) Check Email ")
print("2) Display Dataset ")
print("3) Display Classification report ")
print("4) Display Confusion Matrix ")
print("5) Exit ")

choice = int(input("Your Choice: "))

# if choice == 1:
#     user_email = input("Paste your email below: \n")
#     p.predict_email(user_email)
while choice != 5:
    match choice:
        case 1:
            print("Please note, more accuracte scores for longer emails.")
            user_email = input("Paste your email below: \n")
            p.predict_email(user_email)
        case 2:
            t.display_data(t.df)
        case 3:
            t.display_classification()
        case 4:
            t.display_confusion()
        case 5:
            exit
    choice = int(input("Your Choice: "))


# tests
# predict_email("Congratulations! You have won a free prize. Click here right now to claim your reward!\n")
# predict_email("CLICK HERE RIGHT NOW TO WIN A FREE CAR. RIGHT NOW. DON'T WANT TO MISS OUT ON A CHANCE TO WIN A CAR!\n")
# predict_email("Hello, I hope you are doing well. Let me know when you are free to book an appointment to more forward with our proposed deal\n")
# TRY THIS
# This is a scam email, act now to intiate our scam process and we will take all your money.