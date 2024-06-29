from sklearn.inspection import partial_dependence
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import pandas  as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np



selected_feature = pd.read_csv("sorted_best_selected_svr_all.csv").iloc[:,1].tolist()
part_selected_feature = []
for i in range(1):
    part_selected_feature.append(selected_feature[i])

df_train = pd.read_csv("df_exp_train.csv")
x_train = df_train.iloc[:, 1:-1]
y_train = df_train.iloc[:,-1]
x_train = x_train[selected_feature]
print(np.min(x_train.iloc[:,0]))
model = SVR(C=10, gamma=1)
scaler = StandardScaler().fit(x_train)
X_train_std = pd.DataFrame(scaler.transform(x_train))
copy = X_train_std


norms = np.linalg.norm(X_train_std, axis=1, keepdims=True)
normalizer = Normalizer().fit(X_train_std)
X_train_std = pd.DataFrame(normalizer.transform(X_train_std))
X_train_std.columns = selected_feature
model.fit(X_train_std, y_train)




for i in range(7):        

    my_plots = partial_dependence(model,       # column numbers of plots we want to show
                                    X=X_train_std,            # raw predictors data.
                                    features=[selected_feature[i]],
                                    kind="both")
    # print(my_plots)
    plt.figure(figsize=(6, 4))
    # for j in range(100):
    #     plt.scatter(my_plots['values'][0], my_plots['individual'][0][j], s = 5)
    # plt.plot(my_plots['values'][0], my_plots['individual'][0])
    # plt.plot(my_plots['values'][0], my_plots['average'][0])

    # print(scaler.inverse_transform(my_plots['values'][0]))


    inverse_norms = my_plots['values'][0] * norms
    # print(my_plots['values'][0])

    
    # print(np.min(inverse_norms))
    # print(np.max(inverse_norms))
    inverse_norms_std = inverse_norms * scaler.scale_[i] + scaler.mean_[i]
    # print(np.min(inverse_norms_std))
    # print(np.max(inverse_norms_std))

    # plt.plot(my_plots['values'][0], my_plots['average'][0], color = "r", linewidth = 3)
    plt.plot(my_plots['values'][0], my_plots['average'][0], color = "r", linewidth = 3)

    # for j in range(len(my_plots['individual'][0])):
    #     plt.plot(my_plots['values'][0] * scaler.scale_[i] + scaler.mean_[i], my_plots['individual'][0][j], color = "c", alpha = 0.01)
        # plt.plot(my_plots['values'][0], my_plots['individual'][0][j], color = "c", alpha = 0.02)

    # print(my_plots['average'][0])
    # for j in range(2):
    #     plt.plot(my_plots['values'][0], my_plots['individual'][0][j], color = "c", alpha = 1)
    plt.title('Partial Dependence Plot (SVR)')
    plt.xlabel(selected_feature[i])
    plt.ylabel('Partial Dependence')
    plt.ylim(1.7, 2.5)
    
    plt.grid(alpha=.3)
    plt.savefig("pdp_svr_std_all" + str(i) + ".png")