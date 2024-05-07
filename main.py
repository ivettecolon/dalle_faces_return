import pandas as pd
from sklearn.model_selection import train_test_split

from salmon.triplets.offline import OfflineEmbedding

# Read in data
df = pd.read_csv("triplet_fpo_faces_2023-10-03.csv")  # from dashboard
# em = pd.read_csv("embedding.csv")  # from dashboard; optional

X = df[["head", "winner", "loser"]].to_numpy()
X_train, X_test = train_test_split(X, random_state=42, test_size=0.2)

n = int(X.max() + 1)  # number of targets
d = 3  # embed into 2 dimensions

# Fit the model
model = OfflineEmbedding(n=n, d=d, max_epochs=20_000)
# model.initialize(X_train, embedding=em.to_numpy())  # (optional)

model.fit(X_train, X_test)

# Inspect the model
model.embedding_  # embedding
model.history_  # to view information on how well train/test performed

embs = pd.DataFrame(model.embedding_)
embs.to_csv("triplet_fpo_faces_groupEmb_3d.csv")
