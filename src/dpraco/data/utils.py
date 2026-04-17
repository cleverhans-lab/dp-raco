import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dpraco.data.dataloader import GeneralData
import pandas as pd


def one_hot(x, num_class=2):
    # breakpoint()
    return np.eye(num_class)[x]


def split_into_folds(
        features: np.ndarray,
        labels: np.ndarray,
        sensitives: np.ndarray,
        K: int,
        fairness_constraint: str,
        rng: np.random.Generator
):
    num_samples = features.shape[0]

    if fairness_constraint in ["[DemographicParity", "EqualityOfOdds"]:
        combined_stratify = np.array([
            f"{labels[i].argmax()}_{sensitives[i].argmax()}"
            for i in range(num_samples)
        ])
    else:
        combined_stratify = np.array([labels[i].argmax() for i in range(num_samples)])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)

    folds = []
    for train_index, test_index in skf.split(features, combined_stratify):
        folds.append((train_index, test_index))

    return folds


def get_disparity(fairness_metric, preds, sensitives, labels=None, **kwargs):
    if fairness_metric == "DemParity":
        return dem_disparity(preds, sensitives, **kwargs)
    elif fairness_metric == "ErrorParity":
        return error_disparity(preds, sensitives, labels, **kwargs)
    elif fairness_metric == "EqualityOfOdds":
        return equality_of_odds_disparity(preds, sensitives, labels, **kwargs)
    else:
        raise ValueError(f"Unknown fairness metric: {fairness_metric}")


def dem_disparity(
    preds, sensitives, num_sensitives=2, interpretation="max_vs_min", **kwargs
):
    pos_prediction = preds @ one_hot(sensitives, num_class=num_sensitives)
    sensitive_group_count = one_hot(sensitives, num_class=num_sensitives).sum(axis=0)
    prob_pos_prediction_per_subgroup = pos_prediction / sensitive_group_count
    if interpretation == "one_vs_average":
        avg_prob_pos_prediction = pos_prediction.sum() / sensitive_group_count.sum()
        dem_disparity = (
            prob_pos_prediction_per_subgroup - avg_prob_pos_prediction
        ).max()
    elif interpretation == "one_vs_others":
        others_count = sensitive_group_count.sum() - sensitive_group_count
        others_pos_prediction = pos_prediction.sum() - pos_prediction
        others_prob_pos_prediction_per_subgroup = others_pos_prediction / others_count
        dem_disparity = (
            prob_pos_prediction_per_subgroup - others_prob_pos_prediction_per_subgroup
        ).max()
    elif interpretation == "max_vs_min":
        dem_disparity = (
            prob_pos_prediction_per_subgroup.max()
            - prob_pos_prediction_per_subgroup.min()
        )
    else:
        raise ValueError(f"Unknown interpretation: {interpretation}")

    return dem_disparity


def error_disparity(preds, sensitives, labels, **kwargs):
    data = pd.DataFrame(
        np.c_[preds, labels, sensitives], columns=["prediction", "truth", "sensitive"]
    )
    error_per_subgroup = []
    # breakpoint()
    for z in np.unique(sensitives):
        error_per_subgroup.append(
            data.query(f"sensitive == {z} and prediction != truth")["prediction"].mean()
        )

    error_disparity = np.max(error_per_subgroup) - np.min(error_per_subgroup)
    return error_disparity


def equality_of_odds_disparity(preds, sensitives, labels, **kwargs):
    data = pd.DataFrame(
        np.c_[preds, labels, sensitives], columns=["prediction", "truth", "sensitive"]
    )
    Y_set = np.unique(labels)
    Z_set = np.unique(sensitives)

    disparity_list = []
    for z in Z_set:
        for yhat in Y_set:
            for y in Y_set:
                prob_for_z = len(
                    data.query(
                        f"prediction == {yhat} and truth == {y} and sensitive == {z}"
                    )
                ) / len(data.query(f"truth == {y} and sensitive == {z}"))
                prob_but_z = len(
                    data.query(
                        f"prediction == {yhat} and truth == {y} and sensitive != {z}"
                    )
                ) / len(data.query(f"truth == {y} and sensitive != {z}"))
                disparity_list.append(prob_for_z - prob_but_z)

    equality_of_odds_disparity = np.max(disparity_list)
    return equality_of_odds_disparity


def sklearn_disparity(
    fairness_metric, model, features, sensitives, labels=None, **kwargs
):
    if fairness_metric == "DemParity":
        return sklearn_dem_disparity(model, features, sensitives, **kwargs)
    elif fairness_metric == "ErrorParity":
        return sklearn_error_disparity(model, features, sensitives, labels, **kwargs)
    elif fairness_metric == "EqualityOfOdds":
        return sklearn_equality_of_odds_disparity(
            model, features, sensitives, labels, **kwargs
        )
    else:
        raise ValueError(f"Unknown fairness metric: {fairness_metric}")


def sklearn_dem_disparity(
    model, features, sensitives, interpretation="max_vs_min", **kwargs
):
    pos_prediction = model.predict(features) @ one_hot(sensitives, num_class=2)
    sensitive_group_count = one_hot(sensitives, num_class=2).sum(axis=0)
    prob_pos_prediction_per_subgroup = pos_prediction / sensitive_group_count
    if interpretation == "one_vs_average":
        avg_prob_pos_prediction = pos_prediction.sum() / sensitive_group_count.sum()
        dem_disparity = (
            prob_pos_prediction_per_subgroup - avg_prob_pos_prediction
        ).max()
    elif interpretation == "one_vs_others":
        others_count = sensitive_group_count.sum() - sensitive_group_count
        others_pos_prediction = pos_prediction.sum() - pos_prediction
        others_prob_pos_prediction_per_subgroup = others_pos_prediction / others_count
        dem_disparity = (
            prob_pos_prediction_per_subgroup - others_prob_pos_prediction_per_subgroup
        ).max()
    elif interpretation == "max_vs_min":
        dem_disparity = (
            prob_pos_prediction_per_subgroup.max()
            - prob_pos_prediction_per_subgroup.min()
        )
    else:
        raise ValueError(f"Unknown interpretation: {interpretation}")

    return dem_disparity

def sklearn_error_disparity(model, features, sensitives, labels, **kwargs):
    predictions = model.predict(features)
    data = pd.DataFrame(
        np.c_[predictions, labels, sensitives],
        columns=["prediction", "truth", "sensitive"],
    )
    error_per_subgroup = []
    for z in np.unique(sensitives):
        error_per_subgroup.append(
            data.query(f"sensitive == {z} and prediction != truth")["prediction"].mean()
        )

    error_disparity = np.max(error_per_subgroup) - np.min(error_per_subgroup)
    return error_disparity


def sklearn_equality_of_odds_disparity(model, features, sensitives, labels, **kwargs):
    predictions = model.predict(features)
    data = pd.DataFrame(
        np.c_[predictions, labels, sensitives],
        columns=["prediction", "truth", "sensitive"],
    )
    Y_set = np.unique(labels)
    Z_set = np.unique(sensitives)

    disparity_list = []
    for z in Z_set:
        for yhat in Y_set:
            for y in Y_set:
                prob_for_z = len(
                    data.query(
                        f"prediction == {yhat} and truth == {y} and sensitive == {z}"
                    )
                ) / len(data.query(f"truth == {y} and sensitive == {z}"))
                prob_but_z = len(
                    data.query(
                        f"prediction == {yhat} and truth == {y} and sensitive != {z}"
                    )
                ) / len(data.query(f"truth == {y} and sensitive != {z}"))
                disparity_list.append(prob_for_z - prob_but_z)

    equality_of_odds_disparity = np.max(disparity_list)
    return equality_of_odds_disparity


def process_data(np_rng, args, log):
    output_path = f"{args.seed}_{args.dataset}_processed.npz"
    DO_NOT_SKIP = False
    if not os.path.exists(output_path) or DO_NOT_SKIP:
        full_data = GeneralData(
            path=args.path,
            random_state=np_rng,
            sensitive_attributes=args.sensitive_attributes,
            cols_to_norm=args.cols_to_norm,
            output_col_name=args.output_col_name,
            split=args.split,
        )

        dataset_train = full_data.getTrain(return_tensor=False)
        dataset_test = full_data.getTest(return_tensor=False)

        train_features = np.concatenate([x[0][None, :] for x in dataset_train], axis=0)
        train_labels = np.array([x[2] for x in dataset_train])
        train_sensitives = np.array([x[3] for x in dataset_train])

        test_features = np.concatenate([x[0][None, :] for x in dataset_test], axis=0)
        test_labels = np.array([x[2] for x in dataset_test])
        test_sensitives = np.array([x[3] for x in dataset_test])

        log(
            "Train Label Top-Count/All Ratio: {:.4f}",
            np.unique(train_labels, return_counts=True)[1].max() / len(train_labels),
        )

        np.savez(
            output_path,
            train_features=train_features,
            train_labels=train_labels,
            train_sensitives=train_sensitives,
            test_features=test_features,
            test_labels=test_labels,
            test_sensitives=test_sensitives,
        )
        return (
            train_features,
            train_labels,
            train_sensitives,
            test_features,
            test_labels,
            test_sensitives,
        )
    else:
        log("Dataset already processed. Loading from file")
        data = np.load(output_path)
        train_features = data["train_features"]
        train_labels = data["train_labels"]
        train_sensitives = data["train_sensitives"]
        test_features = data["test_features"]
        test_labels = data["test_labels"]
        test_sensitives = data["test_sensitives"]
        return (
            train_features,
            train_labels,
            train_sensitives,
            test_features,
            test_labels,
            test_sensitives,
        )
