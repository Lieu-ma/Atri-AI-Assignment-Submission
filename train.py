
import argparse
import numpy as np
import wandb
from keras.datasets import fashion_mnist

from neural_network import NumpyMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="atri-fashion-mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--optimizer",
        default="rmsprop",
        choices=["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_init", default="xavier", choices=["random", "xavier"])
    parser.add_argument("--activation", default="tanh", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def prepare_data(seed=42):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X_train.shape[0])
    val_size = int(0.10 * X_train.shape[0])

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return (
        X_train[train_idx],
        y_train[train_idx],
        X_train[val_idx],
        y_train[val_idx],
        X_test,
        y_test,
    )


def main():
    args = parse_args()

    run = None
    if not args.no_wandb:
        run = wandb.init(project=args.project, config=vars(args))
        run.name = (
            f"hl_{args.num_hidden_layers}_hs_{args.hidden_size}"
            f"_bs_{args.batch_size}_ac_{args.activation}_opt_{args.optimizer}"
        )

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(seed=args.seed)

    model = NumpyMLP(
        input_size=784,
        hidden_layers=[args.hidden_size] * args.num_hidden_layers,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        seed=args.seed,
    )

    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        wandb_run=run,
        verbose=True,
    )

    train_loss, train_acc = model.evaluate(X_train, y_train, weight_decay=args.weight_decay)
    val_loss, val_acc = model.evaluate(X_val, y_val, weight_decay=args.weight_decay)
    test_loss, test_acc = model.evaluate(X_test, y_test, weight_decay=args.weight_decay)

    print(f"Final train accuracy: {train_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")

    if run is not None:
        run.log(
            {
                "final_train_loss": train_loss,
                "final_train_acc": train_acc,
                "final_val_loss": val_loss,
                "final_val_acc": val_acc,
                "final_test_loss": test_loss,
                "final_test_acc": test_acc,
            }
        )
        run.finish()


if __name__ == "__main__":
    main()
