import sklearn
import torch
import torch.nn.functional as F


def feature_loss(fnn_out, lambda_=0.):
    return lambda_ * (fnn_out ** 2).sum() / fnn_out.shape[1]


def penalized_cross_entropy(logits, truth, fnn_out, feature_penalty=0.):
    return F.binary_cross_entropy_with_logits(logits.view(-1), truth.view(-1)) + feature_loss(fnn_out, feature_penalty)

def penalized_cross_entropy_MutiTask(logits, truth, fnn_out, feature_penalty=0.):
    loss = torch.nn.CrossEntropyLoss()
    return loss(logits, truth.argmax(-1)) + feature_loss(fnn_out, feature_penalty)

def penalized_mse(logits, truth, fnn_out, feature_penalty=0.):
    return F.mse_loss(logits.view(-1), truth.view(-1)) + feature_loss(fnn_out, feature_penalty)


def calculate_metric(logits,
                     truths,
                     regression=True):
    """Calculates the evaluation metric."""
    if regression:
        # root mean squared error
        # return torch.sqrt(F.mse_loss(logits, truths, reduction="none")).mean().item()
        # mean absolute error
        return "MAE", ((logits.view(-1) - truths.view(-1)).abs().sum() / logits.numel()).item()
    elif len(logits.shape) == 1:
        # return sklearn.metrics.roc_auc_score(truths.view(-1).tolist(), torch.sigmoid(logits.view(-1)).tolist())
        return "accuracy", accuracySingle(logits, truths)
    else:
        return "accuracy", accuracyMuti(logits, truths)


def accuracySingle(logits, truths):
    return (((truths.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / truths.numel()).item()

def accuracyMuti(logits, truths):
    return ((logits.argmax(-1)==truths.argmax(-1)).sum()/truths.numel()).item()
