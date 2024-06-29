from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

selected_feature = pd.read_csv("sorted_best_selected_svr.csv").iloc[:,1].tolist()
part_selected_feature = []
for i in range(1):
    part_selected_feature.append(selected_feature[i])

df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]

model = SVR(C=10, gamma=1)

# scaler = StandardScaler().fit(x_train)
# X_train_std = scaler.transform(x_train)
# normalizer = Normalizer().fit(X_train_std)
# X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
# X_train_std.columns = part_selected_feature
model.fit(x_train, y_train)

# plot           

for i in range(5):        
    my_plots = partial_dependence(model,       # column numbers of plots we want to show
                                    X=x_train,            # raw predictors data.
                                    features=[selected_feature[i]],
                                    kind="both")
    # print(my_plots)
    plt.figure(figsize=(6, 4))
  
    # for j in range(100):
    #     plt.scatter(my_plots['values'][0], my_plots['individual'][0][j], s = 5)
    # plt.plot(my_plots['values'][0], my_plots['individual'][0])
    plt.plot(my_plots['values'][0], my_plots['average'][0], color = "r", linewidth = 3)
    for j in range(len(my_plots['individual'][0])):
        plt.plot(my_plots['values'][0], my_plots['individual'][0][j], color = "c", alpha = 0.05)
    plt.title('Partial Dependence Plot (SVR)')
    plt.xlabel(selected_feature[i])
    plt.ylabel('Partial Dependence')
    # plt.ylim(1.83, 2.45)
    plt.ylim(1.7, 2.7)
    plt.grid(alpha=.3)
    plt.savefig("pdp_svr" + str(i) + ".png")