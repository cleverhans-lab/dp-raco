import torch
import torch.utils.data as Data
from torchvision import transforms

import pandas as pd
import random
import numpy as np
from PIL import Image
import os


class GeneralData:
    def __init__(
        self,
        path,
        random_state,
        sensitive_attributes=None,
        cols_to_norm=None,
        split=0.60,
        output_col_name=None,
        skip_sensitive=False,
    ):
        if not sensitive_attributes:
            raise Exception(
                "No Sensitive Attributes Provided. Please provide one or more"
            )

        if not output_col_name:
            raise Exception("No output column name entered. Please provide one")

        self.output_col_name = output_col_name
        self.sensitive_attributes = sorted(sensitive_attributes)

        self.path = path
        df = pd.read_csv(self.path)

        if self.sensitive_attributes:
            non_sens_attr = sorted(
                list(
                    set(df.columns).difference(
                        set(self.sensitive_attributes + [output_col_name])
                    )
                )
            )
        else:
            non_sens_attr = sorted(list(df.columns).difference(set([output_col_name])))

        one_hot_cols = list(set(non_sens_attr).difference(cols_to_norm))
        df = pd.get_dummies(df, columns=one_hot_cols)
        self.non_sens_attr = list(
            set(df.columns).difference(
                set(self.sensitive_attributes + [output_col_name])
            )
        )

        # Splitting Data
        self.df_train = df.sample(frac=split, random_state=random_state)
        self.df_test = df.drop(self.df_train.index)

        self.df_train_idx = self.df_train.index
        self.df_test_idx = self.df_test.index

        if cols_to_norm:
            self.mean_train = self.df_train[cols_to_norm].mean()
            self.std_train = self.df_train[cols_to_norm].std()

            for col in cols_to_norm:
                self.df_train[col] = self.df_train[col].apply(
                    lambda x: (x - self.mean_train[col]) / self.std_train[col]
                )
                self.df_test[col] = self.df_test[col].apply(
                    lambda x: (x - self.mean_train[col]) / self.std_train[col]
                )

        self.skip_sensitive = skip_sensitive

    def getTrain(self, return_tensor=True, skip_sensitive=False):
        return TabularDataset(
            self.df_train,
            self.non_sens_attr,
            self.sensitive_attributes,
            output_col_name=self.output_col_name,
            return_tensor=return_tensor,
            skip_sensitive=self.skip_sensitive,
        )

    def getTest(self, return_tensor=True):
        return TabularDataset(
            self.df_test,
            self.non_sens_attr,
            self.sensitive_attributes,
            output_col_name=self.output_col_name,
            return_tensor=return_tensor,
            skip_sensitive=self.skip_sensitive,
        )

    def calculateP_s(self, demographic_parity=True):
        if demographic_parity:
            dataset = self.getTrain()
            sens = torch.zeros(dataset.count_attr[0])
            for i in range(dataset.__len__()):
                _, u, _, _ = dataset.__getitem__(i)
                sens += u
            sens /= dataset.__len__()
            return torch.diag(1 / (sens) ** 0.5)
        else:
            dataset = self.getTrain()
            diff_matrices = [
                torch.zeros(dataset.count_attr[0]),
                torch.zeros(dataset.count_attr[0]),
            ]
            lengths = [0, 0]
            for i in range(dataset.__len__()):
                _, u, lab, _ = dataset.__getitem__(i)
                diff_matrices[lab] += u
                lengths[lab] += 1
            diff_matrices[0] /= lengths[0]
            diff_matrices[1] /= lengths[1]
            return [
                torch.diag(1 / (diff_matrices[0]) ** 0.5),
                torch.diag(1 / (diff_matrices[1]) ** 0.5),
            ]

    def get_train_test_idx(self):
        return self.df_train_idx, self.df_test_idx


class TabularDataset(Data.Dataset):
    def __init__(
        self,
        df,
        non_sens_attr,
        sensitive_attributes,
        output_col_name,
        return_tensor=True,
        skip_sensitive=False,
    ):
        self.df = df
        self.sensitive_attributes = sensitive_attributes
        self.output_col_name = output_col_name

        self.one_hot_non_senstive = self.df[non_sens_attr]
        self.sensitive_table = self.df[self.sensitive_attributes]
        self.output = self.df[output_col_name]

        self.count_attr = []
        self.attr_no = {}

        for col_name in self.sensitive_attributes:
            self.attr_no[col_name] = {}
            count = 0
            for col_nam_attr in list(self.df[col_name].unique()):
                self.attr_no[col_name][col_nam_attr] = count
                count += 1
            self.count_attr.append(count)

        for i in range(len(self.count_attr) - 2, -1, -1):
            self.count_attr[i] = self.count_attr[i] * self.count_attr[i + 1]
        self.count_attr.append(1)

        self.return_tensor = return_tensor
        self.skip_sensitive = skip_sensitive

    def __len__(self):
        return len(self.one_hot_non_senstive.index)

    def __getitem__(self, idx):
        non_sensitive_attributes = np.array(self.one_hot_non_senstive.iloc[idx])
        sensitive_one_hot, sens_ind = self.onehotlookup(self.sensitive_table.iloc[idx])
        label = self.output.iloc[idx]
        if label == "Yes":
            label = 1
        else:
            label = 0
        non_sensitive_attributes = non_sensitive_attributes.astype(np.float32)
        if self.return_tensor:
            if self.skip_sensitive:
                return torch.from_numpy(non_sensitive_attributes), label
            else:
                return (
                    torch.from_numpy(non_sensitive_attributes),
                    sensitive_one_hot,
                    label,
                    sens_ind,
                )
        else:
            return non_sensitive_attributes, sensitive_one_hot, label, sens_ind

    def onehotlookup(self, df):
        if self.return_tensor:
            one_hot_vector = torch.zeros(self.count_attr[0])
        else:
            one_hot_vector = np.zeros(self.count_attr[0])
        index = 0
        for i, attr in enumerate(self.sensitive_attributes):
            index += self.count_attr[i + 1] * self.attr_no[attr][df[attr]]
        one_hot_vector[index] = 1
        return one_hot_vector, index

    def get_preprocessed_df(self, sensitive_col="z", output_col="y"):
        _df = self.df.copy()

        # for col in self.sensitive_attributes:
        _df[output_col] = _df[self.output_col_name].apply(
            lambda x: 1 if x == "Yes" else 0
        )
        _df[sensitive_col] = self.sensitive_table.apply(
            lambda x: self.onehotlookup(x)[1], axis=1
        )

        _df = _df.drop(self.sensitive_attributes + [self.output_col_name], axis=1)

        return _df


class UTKFaceDataset9C(Data.Dataset):
    def __init__(
        self,
        sensitive_attribute="race",
        split=0.75,
        validation_ratio=0.2,
        kind="train",
        seed=100,
        skip_sensitive=False,
    ):
        self.sens_count = 0
        if sensitive_attribute == "gender":
            self.sens_count = 2
            self.sens_index = 1
        if sensitive_attribute == "race":
            self.sens_count = 5
            self.sens_index = 2
        self.age_ranges = [
            (0, 10),
            (10, 15),
            (15, 20),
            (20, 25),
            (25, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 120),
        ]
        if sensitive_attribute == "age":
            self.sens_count = 9
            self.sens_index = 0
        self.transforms = transforms.Compose(
            [transforms.Resize(128), transforms.ToTensor()]
        )
        self.path = "/mfsnic/datasets/utkface/UTKFace/"
        self.image_names = os.listdir(self.path)
        if seed:
            random.Random(seed).shuffle(self.image_names)
        if kind == "train":
            self.dataset = self.image_names[
                : int(split * (1 - validation_ratio) * len(self.image_names))
            ]
        elif kind == "valid":
            self.dataset = self.image_names[
                int(split * (1 - validation_ratio) * len(self.image_names)) : int(
                    split * len(self.image_names)
                )
            ]
        elif kind == "test":
            self.dataset = self.image_names[int(split * len(self.image_names)) :]
        else:
            raise NotImplementedError

        self.skip_sensitive = skip_sensitive

    def __getitem__(self, idx):
        image_retr = self.dataset[idx]
        attributes = image_retr.split("_")[:-1]

        sensitive_one_hot = torch.zeros(self.sens_count)
        sens_ind = int(attributes[self.sens_index])
        sensitive_one_hot[sens_ind] = 1

        label = 0
        for i, (a, b) in enumerate(self.age_ranges):
            if a < int(attributes[0]) and int(attributes[0]) <= b:
                label = i
                break

        image_vec = Image.open(self.path + image_retr)
        u = self.transforms(image_vec)
        if self.skip_sensitive:
            return u, label
        else:
            return u, sensitive_one_hot, label, sens_ind

    def __len__(self):
        return len(self.dataset)


class UTKFaceDataset(Data.Dataset):
    def __init__(
        self,
        sensitive_attribute="race",
        split=0.75,
        train=True,
        seed=100,
        data_path="./Datasets/UTKFace/UTKFace/",
        use_fairpate_split=False,
    ):
        self.sens_count = 0
        if sensitive_attribute == "gender":
            self.sens_count = 2
            self.sens_index = 1
        if sensitive_attribute == "race":
            self.sens_count = 5
            self.sens_index = 2
        self.age_ranges = [
            (0, 10),
            (10, 15),
            (15, 20),
            (20, 25),
            (25, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 120),
        ]
        if sensitive_attribute == "age":
            self.sens_count = 9
            self.sens_index = 0

        self.transforms = transforms.Compose(
            [transforms.Resize(128), transforms.ToTensor()]
        )
        self.path = data_path
        self.use_fairpate_split = use_fairpate_split

        if self.use_fairpate_split:
            all_files = np.load("/mfsnic/projects/fairPATE/utkface_files.npy")
            self.image_names = [
                os.path.join(self.path, img_name) for img_name in all_files
            ]
            # for syncing with other models
            self.train_indices = np.arange(len(self.image_names))[:-1500]
            self.test_indices = np.arange(len(self.image_names))[-1500:]

            if train:
                self.dataset = self.image_names[:-1500]
            else:
                self.dataset = self.image_names[-1500:]
        else:
            self.image_names = os.listdir(self.path)
            if seed:
                random.Random(seed).shuffle(self.image_names)

            if train:
                self.dataset = self.image_names[: int(split * len(self.image_names))]
            else:
                self.dataset = self.image_names[int(split * len(self.image_names)) :]

    def __getitem__(self, idx):
        image_retr = self.dataset[idx]
        attributes = image_retr.split("_")[:-1]

        sensitive_one_hot = torch.zeros(self.sens_count)
        sens_ind = int(attributes[self.sens_index])
        sensitive_one_hot[sens_ind] = 1

        label = int(attributes[1])  # gender

        image_vec = Image.open(self.path + image_retr)
        u = self.transforms(image_vec)

        return u, sensitive_one_hot, label, sens_ind

    def __len__(self):
        return len(self.dataset)
