import os
import tqdm
import copy
import random
import logging
from absl import app
from absl import flags
from torch.utils.data import TensorDataset, DataLoader

import nam.metrics
import nam.data_utils
from nam.model import *

FLAGS = flags.FLAGS

flags.DEFINE_integer("training_epochs", 10, "The number of epochs to run training for.")
flags.DEFINE_float("learning_rate", 1e-3, "Hyperparameter: learning rate.")
flags.DEFINE_float("output_regularization", 0.0, "Hyperparameter: feature reg")
flags.DEFINE_float("l2_regularization", 0.0, "Hyperparameter: l2 weight decay")
flags.DEFINE_integer("batch_size", 32, "Hyperparameter: batch size.")
flags.DEFINE_string("log_file", None, "File where to store summaries.")
flags.DEFINE_string("dataset", "BreastCancer", "Name of the dataset to load for training.")
flags.DEFINE_float("decay_rate", 0.995, "Hyperparameter: Optimizer decay rate")
flags.DEFINE_float("dropout", 0.5, "Hyperparameter: Dropout rate")
flags.DEFINE_integer("data_split", 1, "Dataset split index to use. Possible values are 1 to `FLAGS.num_splits`.")
flags.DEFINE_integer("seed", 1, "Seed used for reproducibility.")
flags.DEFINE_float("feature_dropout", 0.0, "Hyperparameter: Prob. with which features are dropped")
flags.DEFINE_integer("n_basis_functions", 1000, "Number of basis functions to use in a FeatureNN for a real-valued feature.")
flags.DEFINE_integer("units_multiplier", 2, "Number of basis functions for a categorical feature")
flags.DEFINE_integer("n_models", 1, "the number of models to train.")
flags.DEFINE_integer("n_splits", 3, "Number of data splits to use")
flags.DEFINE_integer("id_fold", 1, "Index of the fold to be used")
flags.DEFINE_list("hidden_units", [], "Amounts of neurons for additional hidden layers, e.g. 64,32,32")
flags.DEFINE_string("shallow_layer", "exu", "Activation function used for the first layer: (1) relu, (2) exu")
flags.DEFINE_string("hidden_layer", "relu", "Activation function used for the hidden layers: (1) relu, (2) exu")
flags.DEFINE_boolean("regression", False, "Boolean for regression or classification")
flags.DEFINE_integer("early_stopping_epochs", 60, "Early stopping epochs")
_N_FOLDS = 5


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(x_train, y_train, x_validate, y_validate, device):
    model = NeuralAdditiveModel(
        input_size=x_train.shape[-1],
        # feature size, 0 is sample and 1 is the feature, this is one iter of torch dataloader
        output_size=1 if len(y_train.shape)==1 else y_train.shape[-1],
        shallow_units=nam.data_utils.calculate_n_units(x_train, 1000, 2),
        # for feature network, it is changing with data and I am not sure why
        hidden_units=list(map(int, [])),  # for feature network
        shallow_layer=ExULayer,  # special operational layer designed for this model
        hidden_layer=ExULayer,
        hidden_dropout=0.3,
        feature_dropout=0.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=FLAGS.learning_rate,
                                  weight_decay=FLAGS.l2_regularization)
    criterion = nam.metrics.penalized_mse if FLAGS.regression else nam.metrics.penalized_cross_entropy
    if FLAGS.regression:
        criterion = nam.metrics.penalized_mse
    elif len(y_train.shape)==1:
        nam.metrics.penalized_cross_entropy
    else:
        nam.metrics.penalized_cross_entropy_MutiTask
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    validate_dataset = TensorDataset(torch.tensor(x_validate), torch.tensor(y_validate))
    validate_loader = DataLoader(validate_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    n_tries = FLAGS.early_stopping_epochs
    best_validation_score, best_weights = 0, None

    for epoch in range(FLAGS.training_epochs):
        model = model.train()
        total_loss = train_one_epoch(model, criterion, optimizer, train_loader, device)
        logging.info(f"epoch {epoch} | train | {total_loss}")

        scheduler.step()

        model = model.eval()
        metric, val_score = evaluate(model, validate_loader, device)
        logging.info(f"epoch {epoch} | validate | {metric}={val_score}")

        # early stopping
        if val_score <= best_validation_score and n_tries > 0:
            n_tries -= 1
            continue
        elif val_score <= best_validation_score:
            logging.info(f"early stopping at epoch {epoch}")
            break
        best_validation_score = val_score
        best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    pbar = tqdm.tqdm(enumerate(data_loader, start=1), total=len(data_loader))
    total_loss = 0
    for i, (x, y) in pbar:
        x, y = x.to(device), y.to(device)
        logits, fnns_out = model.forward(x)
        loss = criterion(logits, y, fnns_out, feature_penalty=FLAGS.output_regularization)
        total_loss -= (total_loss / i) - (loss.item() / i)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"train | loss = {total_loss:.5f}")
    return total_loss


def evaluate(model, data_loader, device):
    total_score = 0
    metric = None
    for i, (x, y) in enumerate(data_loader, start=1):
        x, y = x.to(device), y.to(device)
        logits, fnns_out = model.forward(x)
        metric, score = nam.metrics.calculate_metric(logits, y, regression=FLAGS.regression)
        total_score -= (total_score / i) - (score / i)
    return metric, total_score


def main(args):
    seed_everything(FLAGS.seed)

    handlers = [logging.StreamHandler()]
    if FLAGS.log_file:
        handlers.append(logging.FileHandler(FLAGS.log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s",
                        handlers=handlers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("load data")
    train, (x_test, y_test) = nam.data_utils.create_test_train_fold(dataset=FLAGS.dataset,
                                                                id_fold=FLAGS.id_fold,
                                                                n_folds=_N_FOLDS,
                                                                n_splits=FLAGS.n_splits,
                                                                regression=not FLAGS.regression)
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    logging.info("begin training")
    test_scores = []
    while True:
        try:
            (x_train, y_train), (x_validate, y_validate) = next(train)
            model = train_model(x_train, y_train, x_validate, y_validate, device)
            metric, score = evaluate(model, test_loader, device)
            test_scores.append(score)
            logging.info(f"fold {len(test_scores)}/{FLAGS.n_splits} | test | {metric}={test_scores[-1]}")
        except StopIteration:
            break

        logging.info(f"mean test score={test_scores[-1]}")


if __name__ == "__main__":
    app.run(main)
