import pandas as pd
import numpy as np
import random
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics import normalized_mutual_info_score as nmi
from pathlib import Path

  

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Adult data sets
training_data = pd.read_csv('adult.data', header = None, names = columns, na_values= ' ?')
test_data = pd.read_csv('adult.test', header = None, names = columns, skiprows = 1, na_values = ' ?')

print("\nMissing  for Training Data")
print(training_data.isnull().sum())

print("\nMissing Values for Testing Data")
print(test_data.isnull().sum())

#1.

# Drop missing values
cleaned_training = training_data.dropna()
cleaned_test = test_data.dropna()

# Get rid of any whitespace that we may have
cleaned_training = cleaned_training.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
cleaned_test = cleaned_test.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
cleaned_test['income'] = cleaned_test['income'].str.rstrip('.')


print("\nCleaned Values for Training Data")
print(cleaned_training.isnull().sum())

print("\nCleaned Values for Testing Data")
print(cleaned_test.isnull().sum())

#2. 

numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
print(cleaned_training.columns)

cleaned_training[numeric_columns] = cleaned_training[numeric_columns].apply(pd.to_numeric, errors = 'raise')


categories = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

numerical_training_data = pd.get_dummies(cleaned_training, columns = categories)
numerical_test_data = pd.get_dummies(cleaned_test, columns = categories)
print(numerical_training_data.dtypes)
print(numerical_test_data.dtypes)

# Wine Datasets

red_wine = pd.read_csv('winequality-red.csv', sep = ';')
white_wine = pd.read_csv('winequality-white.csv', sep = ';')

#1

class KNN:

    def __init__(self):
        self.centroids = None
        self.classes = None

    def getgroup(self, X: pd.DataFrame, Y: pd.Series):
        # Group rows by their label, which is Y in this case
        group = X.groupby(Y)
        self.classes = np.array(sorted(group.groups.keys()))
        self.centroids = group.mean().loc[self.classes].to_numpy()

    @staticmethod
    def squared_distance(D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
        new_D1 = (D1 * D1).sum(axis = 1).reshape(-1, 1) # Get squared matrix, collapse each row, turn into column vec
        new_D2 = (D2 * D2).sum(axis = 1).reshape(1, -1) # Turn into row vec
        new_dist = new_D1 + new_D2 - 2.0 * (D1 @ D2.T)
        return new_dist

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Fit needs to be called before we can make a prediction.")
        
        np_X = X.to_numpy()
        distances = self.squared_distance(np_X, self.centroids)
        index = distances.argmin(axis = 1)
        return self.classes[index]

    def evaluate(self, X: pd.DataFrame, Y: pd.Series) -> float:
        predictions = self.predict(X)
        correct = (predictions == Y.to_numpy()).sum()
        return correct / len(Y)

#2

def zscore(training: pd.DataFrame, testing: pd.DataFrame):
    mean = training.mean()
    std_dev = training.std().replace(0, 1) # Replace 0 std_dev with 1 just in case, we don't want to divide by 0
    return (training - mean) / std_dev, (testing - mean) / std_dev

def split_acc(dataframe: pd.DataFrame, pct_training: int, rng: random.Random):
    index = list(dataframe.index)

    # Randomly reorder our list
    rng.shuffle(index)        

    # Get the amount of training rows                            
    n_train = int(round(len(dataframe) * pct_training / 100.0))
    
    # The first n rows are for training, the rest are for testing
    train_index = index[:n_train]
    test_index = index[n_train:]

    # Get our subsets of the dataframe
    train_set = dataframe.loc[train_index]
    test_set  = dataframe.loc[test_index]

    X_train, y_train = (train_set.drop('quality', axis=1),train_set['quality'],)
    X_test, y_test = (test_set.drop('quality', axis=1), test_set['quality'],)

    # Normalize our data  
    X_train, X_test = zscore(X_train, X_test)

    clf = KNN()
    clf.getgroup(X_train, y_train)
    return clf.evaluate(X_test, y_test)


def run_all_splits(path: Path, splits = (20, 60, 90), seed = 42, n_runs = 5):
    dataframe = pd.read_csv(path, sep=';')            # cleaned data from previous proj.
    rng_master = random.Random(seed)

    acc = {}
    for pct in splits:
        scores = []
        for run in range(n_runs):
            rng = random.Random(rng_master.randint(0, 2**31 - 1))
            scores.append(split_acc(dataframe, pct, rng))
           #print("sum is", sum(scores))
            #print(len(scores))
        acc[pct] = sum(scores) / len(scores)
    return acc

#4

def k_means(X: np.ndarray, k: int, rng: random.Random, iteration: int = 300, tol: float = 1e-4):
    n, d = X.shape
    num_centroids = X[rng.sample(range(n), k), :]

    for _ in range (iteration):
        labels = KNN.squared_distance(X, num_centroids).argmin(axis = 1)
        new_centroids = []
        for c in range(k):
            members = X[labels == c]
            if len(members) == 0:
                # simple fix: reinitialize to a random point
                new_centroids.append(X[rng.randrange(n)])
            else:
                new_centroids.append(members.mean(axis=0))
        new_centroids = np.vstack(new_centroids)

        shift = np.linalg.norm(new_centroids - num_centroids)
        num_centroids = new_centroids
        if shift < tol:
            break
    return labels, num_centroids

def merge_closest(C: np.ndarray):
    k = len(C)
    distance_matrix = KNN.squared_distance(C, C)
    np.fill_diagonal(distance_matrix, np.inf)
    i, j = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
    unchanged = [c for c in range(k) if c not in (i, j)]
    new_merged = (C[i] + C[j]) / 2.0
    new_centroid = np.vstack([new_merged, C[unchanged]])
    return new_centroid, unchanged, (i, j)

def modified_kmeans(X: np.ndarray, Y: np.ndarray, rng: random.Random, k_max: int = 15, k_min: int = 5, min_size: int = 20):
    labels, C = k_means(X, k_max, rng)
    k = k_max
    while k > k_min:
        counts = Counter(labels)
        small = [c for c, sz in counts.items() if sz < min_size]

        if not small:
            break                    # all clusters big enough

        # pick the tiniest cluster and its nearest neighbour
        c_small = min(small, key=counts.get)
        dists   = KNN.squared_distance(C[c_small:c_small+1], C).ravel()
        dists[c_small] = np.inf
        c_merge = dists.argmin()

        c1 = C[c_small].copy()
        c2 = C[c_merge].copy()

        # delete them from C
        C = np.delete(C, [c_small, c_merge], axis=0)

        # now append their midpoint
        C = np.vstack([C, (c1 + c2) / 2.0])
        k -= 1

        # rerun k‑means with fresh centroids
        labels, C = k_means(X, k, rng, iteration=50)

    nmi_val = nmi(Y, labels, average_method = "arithmetic")
    return labels, C, k, nmi_val

def run_modified_kmeans(path: Path, seed: int = 0):
    df  = pd.read_csv(path, sep=';')
    X   = df.drop('quality', axis=1).to_numpy(dtype=float)
    y   = df['quality'].to_numpy()

    rng = random.Random(seed)
    np.random.seed(seed)           # keep NumPy and std‑lib in sync

    labels, C, k_final, nmi_val = modified_kmeans(X, y, rng)
    return k_final, nmi_val



if __name__ == "__main__":
    red_file   = Path("winequality-red.csv")
    white_file = Path("winequality-white.csv")

    red_acc   = run_all_splits(red_file)
    white_acc = run_all_splits(white_file)

    print("Nearest‑Centroid Accuracy (average of 5 random splits)\n")
    print("  Red‑wine  :", red_acc)
    print("  White‑wine:", white_acc)

    print("\nModified k‑means results (k_max = 15, k_min = 5):")
    k_red,  nmi_red  = run_modified_kmeans(red_file,  seed=42)
    k_white,nmi_white= run_modified_kmeans(white_file,seed=42)

    print(f"  Red‑wine  : final k = {k_red:2d},  NMI = {nmi_red:.3f}")
    print(f"  White‑wine: final k = {k_white:2d},  NMI = {nmi_white:.3f}")


#6 

train_for_X = numerical_training_data.drop(columns = "income").copy()
test_for_X = numerical_test_data.drop(columns = "income").copy()

# Get true labels
train_for_Y = (cleaned_training['income'] == '>50K').astype(int)
test_for_Y = (cleaned_test['income'] == '>50K').astype(int)
#print(cleaned_training['income'].unique())   
#print(train_for_Y.unique(), len(train_for_Y))  
test_for_X = test_for_X.reindex(columns=train_for_X.columns, fill_value=0)


# Drop dummy columns
columns_for_income = [col for col in train_for_X if col.startswith('income_')]
train_for_X = train_for_X.drop(columns = columns_for_income)
test_for_X = test_for_X.drop(columns = columns_for_income)


adult_clf = KNN()
adult_clf.getgroup(train_for_X, train_for_Y)
adult_acc = adult_clf.evaluate(test_for_X, test_for_Y)

print(f"Classifying on Adult Dataset, Adult (>50 & <= 50k accuracy) = {adult_acc:.4f} ")