from typing import List
from bcrypt import re
import numpy as np


class Kid:
    def __init__(self, values, parent_values) -> None:
        self.values = values
        self.parent_values = parent_values
        self.probabilities = np.zeros(shape=(parent_values, values))


class Parent:
    def __init__(self, values) -> None:
        self.values = values
        self.probabilities = np.zeros(shape=(values))


class Bayesian_Classifier:
    def __init__(self, parent: Parent, kids: List[Kid]) -> None:
        self.parent = parent
        self.kids = []
        self.kids.extend(kids)

    def predict(self, observations):
        predictions = []
        for observation in observations:
            prob = []
            for val in range(self.parent.values):
                curr_prob = self.parent.probabilities[val]
                for i in range(len(observation)):
                    curr_prob *= self.kids[i].probabilities[val][int(observation[i])]
                prob.append(curr_prob)
            predictions.append(prob.index(max(prob)))
        return predictions

    def train(self, train_data, train_labels):
        for i in range(self.parent.values):
            good = [[0 for _ in range(self.kids[j].values)] for j in range(len(self.kids))]
            good.append(0)
            for j in range(len(train_data)):
                if train_labels[j] == i:
                    good[-1] += 1
                    for k in range(len(train_data[0])):
                        good[k][int(train_data[j][k])] += 1
            self.parent.probabilities[i] = good[-1] / len(train_data)
            for k in range(len(good)-1):
                for v in range(len(good[k])):
                    self.kids[k].probabilities[i][v] = (good[k][v]+1)/(good[-1]+len(good[k]))


    def train_cross_validate(self, full_data, full_labels, n):
        scores = []
        p_len = int(len(full_data)/n)
        for i in range(n):
            val_data = full_data[i*p_len:(i+1)*p_len:]
            val_labels = full_labels[i*p_len:(i+1)*p_len:]
            train_data = full_data[:i*p_len:]
            train_data.extend(full_data[(i+1)*p_len::])
            train_labels = full_labels[:i*p_len:]
            train_labels.extend(full_labels[(i+1)*p_len::])
            self.train(train_data, train_labels)
            pred = self.predict(val_data)
            scores.append(self.evaluate(pred, val_labels))
        self.train(full_data, full_labels)
        return scores

    def evaluate(self, predictions, real):
        correct = 0
        for i in range(len(real)):
            if predictions[i] == real[i]:
                correct += 1
        return correct/len(real)
