from folktables import ACSDataSource, ACSEmployment
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dpraco.data.utils import split_into_folds
from dpraco.data.utils import one_hot


def get_folkstable(cfg, permute_train=False, seed=0):
    
    
    data_source = ACSDataSource(survey_year=cfg.dataset.survey_year, 
                                horizon=cfg.dataset.horizon, 
                                survey=cfg.dataset.survey, 
                                root_dir=cfg.dataset.data_path)
    
    acs_data = data_source.get_data(states=cfg.dataset.states, download=True)
    _features, _labels, _sensitives = ACSEmployment.df_to_numpy(acs_data)
    _features = StandardScaler().fit_transform(_features)

    cfg.dataset.num_classes = len(np.unique(_labels))
    cfg.dataset.num_fairness_classes = len(np.unique(_sensitives))
    cfg.dataset.num_samples, cfg.dataset.num_features = tuple(_features.shape)

    _labels = one_hot(_labels.astype(int), num_class=cfg.dataset.num_classes)
    _sensitives = (_sensitives - 1).astype(int) # make sensitives zero-indexed
    _sensitives = one_hot(_sensitives, num_class=cfg.dataset.num_fairness_classes)

    features, test_features, labels, test_labels, sensitives, test_sensitives = train_test_split(
        _features, _labels, _sensitives, train_size=0.75, random_state=seed)
    
    cfg.dataset.num_train_samples = labels.shape[0]

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