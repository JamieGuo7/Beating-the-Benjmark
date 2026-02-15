import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


class BasePreprocessor:
    def fit_transform(self, X_train, y_train):
        raise NotImplementedError

    def transform(self, X, y=None):
        raise NotImplementedError

    def inverse_transform_y(self, y_scaled):
        raise NotImplementedError



class SequencePreprocessor(BasePreprocessor):
    def __init__(self, window = 30, pca_variance = 0.95):
        self.window = window
        self.pca_variance = pca_variance
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.pca = PCA(n_components = self.pca_variance)
        self.n_features_original = None
        self.n_components = None

    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(self.window, len(X)):
            Xs.append(X[i - self.window:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def fit_transform(self, X_train, y_train):
        """ Fits scalers and PCA on training data. """
        self.n_features_original = X_train.shape[2]

        # Flatten for scaling
        X_train_flat = X_train.reshape(-1, self.n_features_original)

        # Scale features
        X_train_scaled_flat = self.scaler_X.fit_transform(X_train_flat)

        # Apply PCA
        X_train_pca_flat = self.pca.fit_transform(X_train_scaled_flat)
        self.n_components = X_train_pca_flat.shape[1]

        # Reshape back to sequences
        X_train_scaled = X_train_pca_flat.reshape(-1, self.window, self.n_components)

        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))

        print(f"   PCA: {self.n_features_original} → {self.n_components} components")
        return X_train_scaled, y_train_scaled

    def transform(self, X, y=None):
        """ Transform validation/test data using fitted scalers. """
        X_flat = X.reshape(-1, self.n_features_original)
        X_scaled_flat = self.scaler_X.transform(X_flat)
        X_pca_flat = self.pca.transform(X_scaled_flat)
        X_scaled = X_pca_flat.reshape(-1, self.window, self.n_components)

        if y is not None:
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1))
            return X_scaled, y_scaled
        return X_scaled

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


class StandardPreprocessor(BasePreprocessor):
    """ For non-sequential models. """

    def __init__(self, use_pca=False, pca_variance=0.95):
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.pca = PCA(n_components=pca_variance) if use_pca else None

    def fit_transform(self, X_train, y_train):
        """ Scale (and option to apply PCA) to training data. """
        X_scaled = self.scaler_X.fit_transform(X_train)

        if self.use_pca:
            X_scaled = self.pca.fit_transform(X_scaled)
            print(f"   PCA: {X_train.shape[1]} → {X_scaled.shape[1]} components")

        y_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        return X_scaled, y_scaled

    def transform(self, X, y=None):
        X_scaled = self.scaler_X.transform(X)

        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)

        if y is not None:
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1))
            return X_scaled, y_scaled
        return X_scaled

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()