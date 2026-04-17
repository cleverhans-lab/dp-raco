# Train test split

from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from dpraco.data.utils import split_into_folds
from dpraco.data.utils import one_hot




def get_heart(cfg: DictConfig, permute_train=False, seed=0):
    data_path = Path(cfg.dataset.data_path)
    data_path.mkdir(exist_ok=True, parents=True)
    path = data_path / "heart_disease_health_indicators_BRFSS2015.csv"
    
    if not path.exists():
        data = pd.read_csv("https://www.kaggle.com/api/v1/datasets/download/alexteboul/heart-disease-health-indicators-dataset", compression="zip")
        data.to_csv(path, index=False)
    else:
        data = pd.read_csv(path)    
    cfg.dataset.num_fairness_classes = (2 if cfg.dataset.sensitive_feature == "Sex" 
                                     else 6 if cfg.dataset.sensitive_feature == "Education" else None)

    data = data.rename(columns = {"HeartDiseaseorAttack":"target"})
    data["target"] = data.target.astype(int)

    features = data.drop(["target"],axis = 1).values
    labels = data.target.values.ravel()
    labels = one_hot(labels, num_class=cfg.dataset.num_classes).astype(int)

    sensitives = data[cfg.dataset.sensitive_feature].astype(int)
    sensitives = sensitives.values.ravel()
    sensitives = one_hot(sensitives, num_class=cfg.dataset.num_fairness_classes).astype(int)

    features, test_features, labels, test_labels, sensitives, test_sensitives = \
             train_test_split(features, labels, sensitives, test_size = 0.3, random_state = seed)

    # Folds
    K = cfg.training_params.num_folds
    fairness_constraint = cfg.algorithm.constraint_type
    folds = split_into_folds(
        features=features,
        labels=labels,
        sensitives=sensitives,
        K=K,
        fairness_constraint=fairness_constraint,
        rng=np.random.default_rng(cfg.dataset.rng)
    )

    if hasattr(cfg.training_params, "seeds"):
        seeds = cfg.training_params.seeds
        idx = seeds.index(seed)
    else:
        idx = 0

    train_index, val_index = folds[idx]
    
    train_features = features[train_index]
    train_labels = labels[train_index]
    train_sensitives = sensitives[train_index]
    
    val_features = features[val_index]
    val_labels = labels[val_index]
    val_sensitives = sensitives[val_index]

    # per-fold scaling
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)
    
    # set data features
    cfg.dataset.num_train_samples = train_features.shape[0]


    return (
        train_features,
        train_labels,
        train_sensitives,
        val_features,
        val_labels,
        val_sensitives,
        test_features,
        test_labels,
        test_sensitives
    )