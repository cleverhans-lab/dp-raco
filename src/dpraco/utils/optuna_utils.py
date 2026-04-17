import pandas as pd
import yaml


def load_yaml_results(path: str) -> pd.DataFrame:
    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)
    data = yaml_data["solutions"]
    df = pd.DataFrame(data)
    df = pd.concat(
        [
            df.apply(lambda x: x["params"], axis=1, result_type="expand"),
            df.apply(
                lambda x: dict(
                    test_accuracy=x["values"][0], test_constraint=x["values"][1]
                ),
                result_type="expand",
                axis=1,
            ),
        ],
        axis=1,
    )
    df["test_error"] = 1 - df["test_accuracy"]
    return df
