from sklearn.utils import shuffle
from sklearn.datasets import load_wine
from Bayesian_Classifier import Bayesian_Classifier, Kid, Parent
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    bins = 2
    test_size = 0.2
    n_validation = 4

    dataset = load_wine()
    data = dataset.data
    labels = dataset.target
    for i in range(len(data[0])):
        column = [data[j][i] for j in range(len(data))]
        max_val = max(column)
        min_val = min(column)
        for j in range(len(data)):
            if data[j][i] < min_val + ((max_val - min_val)/bins):
                data[j][i] = 0
            elif data[j][i] >= min_val + ((bins-1) * (max_val - min_val)/bins):
                data[j][i] = bins-1
            else:
                for k in range(1, bins-1):
                    if data[j][i] < min_val + ((k+1) * (max_val - min_val)/bins) and data[j][i] >= min_val + (k * (max_val - min_val)/bins):
                        data[j][i] = k
                        break
    full_data = tuple(zip(data, labels))
    shuffled = shuffle(full_data)
    res = list(zip(*shuffled))
    data = res[0]
    labels = res[1]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

    wine_t = Parent(3)
    kids = [Kid(bins, 3) for _ in range(len(data[0]))]
    classifier = Bayesian_Classifier(wine_t, kids)
    scores = classifier.train_cross_validate(X_train, y_train, n_validation)
    pred = classifier.predict(X_test)
    score = classifier.evaluate(pred, y_test)
    print("Validation scores: ", scores, "\nMean validation score: ", sum(scores)/len(scores), "\nTest score: ", score)
