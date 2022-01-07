# Code snippet adjusted from:
# https://www.adriangb.com/scikeras/stable/quickstart.html#training-a-model
import numpy as np
from experiment_impact_tracker.compute_tracker import ImpactTracker
from scikeras.wrappers import KerasClassifier
from sklearn.datasets import make_classification
from tensorflow import keras


def track_neural_network_fit(log_directory="tmp"):
    # launch tracker
    tracker = ImpactTracker(log_directory)
    tracker.launch_impact_monitor()

    # simulate data anf fit network
    X, y = simulate_data(n_samples=10_000, n_features=100)

    model = KerasClassifier(
        get_model,
        loss="sparse_categorical_crossentropy",
        n_layers=2,
        hidden_layer_dim=20,
    )
    model.fit(X, y)

    # check for errors in tracker
    tracker.get_latest_info_and_check_for_errors()
    tracker.stop()


def simulate_data(n_samples, n_features, random_state=0):
    X, y = make_classification(
        n_samples, n_features, n_informative=10, random_state=random_state
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    return X, y


def get_model(n_layers, hidden_layer_dim, meta):
    # note that meta is a special argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))

    for _ in range(n_layers):
        model.add(keras.layers.Dense(hidden_layer_dim))
        model.add(keras.layers.Activation("relu"))

    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model
