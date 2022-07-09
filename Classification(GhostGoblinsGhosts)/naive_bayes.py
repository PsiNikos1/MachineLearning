import numpy as np
import pandas as pd


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)

        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)

        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for i, c in enumerate(self._classes):
            X_c = X[y == c]

            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._priors[i] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict_class(x) for x in X]
        return np.array(y_pred)

    def _predict_class(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # return class with highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        arithmitis = np.exp(-((x - mean) ** 2) / (2 * var))
        paranomastis = np.sqrt(2 * np.pi * var)
        return arithmitis / paranomastis

def load_dataset(csv_path: str, type: str):
    dataset = pd.read_csv(csv_path)
    cols = dataset.columns
    cols = [col for col in cols]

    colour_id_list = [colour_id for colour_id in dataset.drop_duplicates(subset='color')['color']]

    dataset['color'].replace(
        to_replace=[colour_id_list[0], colour_id_list[1], colour_id_list[2], colour_id_list[3], colour_id_list[4],
                    colour_id_list[5]],
        value=[1, 2, 3, 4, 5, 6], inplace=True)
    if type == "train":
        dataset['type'].replace(to_replace=["Ghoul", "Goblin", "Ghost"], value=[0, 1, 2], inplace=True)

    return dataset

def change_predictions_to_monsters_csv (predicted, test_df):
    predicted_lst = [[test_df.loc[idx]['id'].astype(int), el] for idx, el in enumerate(predicted)]

    for i in range(len(predicted_lst)):
        if predicted_lst[i][1] == 0:
            predicted_lst[i][1] = "Ghoul"
        if predicted_lst[i][1] == 1:
            predicted_lst[i][1] = "Goblin"
        if predicted_lst[i][1] == 2:
            predicted_lst[i][1] = "Ghost"

    prediction_df = pd.DataFrame(predicted_lst, columns=['id', 'type'])
    prediction_df.to_csv("naivebayes_output.csv", index=False)

def main():
    df = load_dataset("train.csv", "train")
    X_train = df[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul' ]].to_numpy()
    y_train = df['type'].to_numpy()
    test_df = load_dataset("test.csv", "test")
    X_test = test_df[['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']].to_numpy()

    naivebayes = NaiveBayes()
    naivebayes.fit(X_train, y_train)
    predictions = naivebayes.predict(X_test)
    change_predictions_to_monsters_csv(predictions, test_df)


main()



