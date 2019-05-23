from sklearn.metrics import classification_report
# y_true = [0, 1, 2, 2, 2]
# y_pred = [0, 0, 2, 2, 1]
# target_names = ['class 0', 'class 1', 'class 2']
# print(classification_report(y_true, y_pred, target_names=target_names))

x_true = [0,1,2,1]
x_pred = [0,1,3,3]
target_names = ['EN','EM','ET','EC']
print(classification_report(x_true, x_pred, target_names=target_names))