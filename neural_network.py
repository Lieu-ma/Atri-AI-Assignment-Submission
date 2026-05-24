
import numpy as np


class NumpyMLP:
    def __init__(self, input_size=784, hidden_layers=None, output_size=10,
                 activation="relu", weight_init="xavier", seed=42):
        if hidden_layers is None:
            hidden_layers = [128, 64]

        self.sizes = [input_size] + list(hidden_layers) + [output_size]
        self.num_layers = len(self.sizes) - 1
        self.output_size = output_size
        self.activation_name = activation
        self.weight_init = weight_init
        self.rng = np.random.default_rng(seed)

        self.params = {}
        self._init_params()

        self.velocity = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.square_grad = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def _init_params(self):
        for layer in range(1, self.num_layers + 1):
            n_in = self.sizes[layer - 1]
            n_out = self.sizes[layer]

            if self.weight_init == "xavier":
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = self.rng.uniform(-limit, limit, size=(n_in, n_out))
            elif self.weight_init == "random":
                W = self.rng.standard_normal((n_in, n_out)) * 0.01
            else:
                raise ValueError("weight_init must be 'xavier' or 'random'")

            self.params[f"W{layer}"] = W
            self.params[f"b{layer}"] = np.zeros((1, n_out))

    def _activation(self, z):
        if self.activation_name == "sigmoid":
            z = np.clip(z, -500, 500)
            return 1.0 / (1.0 + np.exp(-z))
        if self.activation_name == "tanh":
            return np.tanh(z)
        if self.activation_name == "relu":
            return np.maximum(0, z)
        raise ValueError("activation must be 'sigmoid', 'tanh', or 'relu'")

    def _activation_derivative(self, z, a):
        if self.activation_name == "sigmoid":
            return a * (1.0 - a)
        if self.activation_name == "tanh":
            return 1.0 - a ** 2
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        raise ValueError("activation must be 'sigmoid', 'tanh', or 'relu'")

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y):
        y = y.astype(int)
        return np.eye(self.output_size)[y]

    def forward(self, X):
        cache = {"A0": X}
        A = X

        for layer in range(1, self.num_layers):
            Z = A @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            A = self._activation(Z)
            cache[f"Z{layer}"] = Z
            cache[f"A{layer}"] = A

        L = self.num_layers
        Z = A @ self.params[f"W{L}"] + self.params[f"b{L}"]
        A = self._softmax(Z)

        cache[f"Z{L}"] = Z
        cache[f"A{L}"] = A

        return A, cache

    def compute_loss(self, y_pred, y_true, weight_decay=0.0):
        m = y_true.shape[0]
        Y = self._one_hot(y_true)
        y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)

        loss = -np.sum(Y * np.log(y_pred)) / m

        l2 = 0.0
        for layer in range(1, self.num_layers + 1):
            l2 += np.sum(self.params[f"W{layer}"] ** 2)

        return loss + (weight_decay / (2 * m)) * l2

    def backward(self, y_pred, y_true, cache, weight_decay=0.0):
        grads = {}
        m = y_true.shape[0]
        Y = self._one_hot(y_true)

        dZ = (y_pred - Y) / m

        for layer in range(self.num_layers, 0, -1):
            A_prev = cache[f"A{layer - 1}"]
            W = self.params[f"W{layer}"]

            grads[f"dW{layer}"] = A_prev.T @ dZ + (weight_decay / m) * W
            grads[f"db{layer}"] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 1:
                dA_prev = dZ @ W.T
                Z_prev = cache[f"Z{layer - 1}"]
                A_prev_hidden = cache[f"A{layer - 1}"]
                dZ = dA_prev * self._activation_derivative(Z_prev, A_prev_hidden)

        return grads

    def _update_params(self, grads, optimizer="sgd", lr=0.001,
                       beta=0.9, beta1=0.9, beta2=0.999, eps=1e-8):
        optimizer = optimizer.lower()

        if optimizer in ["adam", "nadam"]:
            self.t += 1

        for layer in range(1, self.num_layers + 1):
            for name in ["W", "b"]:
                key = f"{name}{layer}"
                grad = grads[f"d{name}{layer}"]

                if optimizer == "sgd":
                    self.params[key] -= lr * grad

                elif optimizer == "momentum":
                    self.velocity[key] = beta * self.velocity[key] + grad
                    self.params[key] -= lr * self.velocity[key]

                elif optimizer == "rmsprop":
                    self.square_grad[key] = beta * self.square_grad[key] + (1 - beta) * (grad ** 2)
                    self.params[key] -= lr * grad / (np.sqrt(self.square_grad[key]) + eps)

                elif optimizer == "adam":
                    self.m[key] = beta1 * self.m[key] + (1 - beta1) * grad
                    self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grad ** 2)

                    m_hat = self.m[key] / (1 - beta1 ** self.t)
                    v_hat = self.v[key] / (1 - beta2 ** self.t)

                    self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

                elif optimizer == "nadam":
                    self.m[key] = beta1 * self.m[key] + (1 - beta1) * grad
                    self.v[key] = beta2 * self.v[key] + (1 - beta2) * (grad ** 2)

                    m_hat = self.m[key] / (1 - beta1 ** self.t)
                    v_hat = self.v[key] / (1 - beta2 ** self.t)
                    nesterov_m = beta1 * m_hat + ((1 - beta1) * grad) / (1 - beta1 ** self.t)

                    self.params[key] -= lr * nesterov_m / (np.sqrt(v_hat) + eps)

                else:
                    raise ValueError(f"Unsupported optimizer: {optimizer}")

    def train_batch(self, X_batch, y_batch, optimizer="sgd", lr=0.001,
                    beta=0.9, beta1=0.9, beta2=0.999, eps=1e-8,
                    weight_decay=0.0):
        optimizer = optimizer.lower()

        if optimizer == "nesterov":
            old_params = {k: v.copy() for k, v in self.params.items()}

            for key in self.params:
                self.params[key] = old_params[key] - lr * beta * self.velocity[key]

            y_pred, cache = self.forward(X_batch)
            loss = self.compute_loss(y_pred, y_batch, weight_decay)
            grads = self.backward(y_pred, y_batch, cache, weight_decay)

            self.params = old_params

            for layer in range(1, self.num_layers + 1):
                for name in ["W", "b"]:
                    key = f"{name}{layer}"
                    grad = grads[f"d{name}{layer}"]
                    self.velocity[key] = beta * self.velocity[key] + grad
                    self.params[key] -= lr * self.velocity[key]

            return loss

        y_pred, cache = self.forward(X_batch)
        loss = self.compute_loss(y_pred, y_batch, weight_decay)
        grads = self.backward(y_pred, y_batch, cache, weight_decay)

        self._update_params(grads, optimizer=optimizer, lr=lr, beta=beta,
                            beta1=beta1, beta2=beta2, eps=eps)

        return loss

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def evaluate(self, X, y, weight_decay=0.0):
        probs, _ = self.forward(X)
        loss = self.compute_loss(probs, y, weight_decay)
        acc = self.accuracy(X, y)
        return loss, acc

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5,
            batch_size=64, optimizer="sgd", lr=0.001, beta=0.9,
            beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
            wandb_run=None, verbose=True):
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n, batch_size):
                end = start + batch_size
                self.train_batch(
                    X_shuffled[start:end],
                    y_shuffled[start:end],
                    optimizer=optimizer,
                    lr=lr,
                    beta=beta,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                )

            train_loss, train_acc = self.evaluate(X_train, y_train, weight_decay)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            log_data = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}

            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, weight_decay)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                log_data["val_loss"] = val_loss
                log_data["val_acc"] = val_acc

            if wandb_run is not None:
                wandb_run.log(log_data)

            if verbose:
                msg = f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f}"
                if X_val is not None and y_val is not None:
                    msg += f" | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                print(msg)

        return history


class LossComparisonMLP(NumpyMLP):
    def __init__(self, *args, loss_type="cross_entropy", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

    def compute_loss(self, y_pred, y_true, weight_decay=0.0):
        m = y_true.shape[0]
        Y = self._one_hot(y_true)
        y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)

        if self.loss_type == "cross_entropy":
            loss = -np.sum(Y * np.log(y_pred)) / m
        elif self.loss_type == "squared_error":
            loss = 0.5 * np.sum((y_pred - Y) ** 2) / m
        else:
            raise ValueError("loss_type must be 'cross_entropy' or 'squared_error'")

        l2 = 0.0
        for layer in range(1, self.num_layers + 1):
            l2 += np.sum(self.params[f"W{layer}"] ** 2)

        return loss + (weight_decay / (2 * m)) * l2

    def backward(self, y_pred, y_true, cache, weight_decay=0.0):
        grads = {}
        m = y_true.shape[0]
        Y = self._one_hot(y_true)

        if self.loss_type == "cross_entropy":
            dZ = (y_pred - Y) / m
        elif self.loss_type == "squared_error":
            dA = y_pred - Y
            inner = np.sum(dA * y_pred, axis=1, keepdims=True)
            dZ = y_pred * (dA - inner) / m
        else:
            raise ValueError("loss_type must be 'cross_entropy' or 'squared_error'")

        for layer in range(self.num_layers, 0, -1):
            A_prev = cache[f"A{layer - 1}"]
            W = self.params[f"W{layer}"]

            grads[f"dW{layer}"] = A_prev.T @ dZ + (weight_decay / m) * W
            grads[f"db{layer}"] = np.sum(dZ, axis=0, keepdims=True)

            if layer > 1:
                dA_prev = dZ @ W.T
                Z_prev = cache[f"Z{layer - 1}"]
                A_prev_hidden = cache[f"A{layer - 1}"]
                dZ = dA_prev * self._activation_derivative(Z_prev, A_prev_hidden)

        return grads
