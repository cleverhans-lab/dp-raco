from pathlib import Path
from argparse import Namespace
import os


class Args(Namespace):
    dataset = "adult"
    num_classes = 2
    output_col_name = "income"
    split = 0.75


def process_arguments(args: Namespace, root_path):
    if args.dataset in ["adult", "credit-card", "parkinsons"]:
        root_path = Path(root_path)
        if args.dataset == "adult":
            path = os.path.join(root_path, "adult_original_purified.csv")
        elif args.dataset == "credit-card":
            path = os.path.join(root_path, "credit-card-defaulters_processed.csv")
        elif args.dataset == "parkinsons":
            path = os.path.join(root_path, "parkinsons_updrs_processed.csv")
        else:
            raise ValueError("Dataset not found")

        if args.dataset == "adult":
            num_inp_attr = 102
        elif args.dataset == "credit-card":
            num_inp_attr = 85
        elif args.dataset == "parkinsons":
            num_inp_attr = 19

        if args.dataset == "adult":
            cols_to_norm = [
                "age",
                "fnlwgt",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            ]
        elif args.dataset == "credit-card":
            cols_to_norm = [
                "LIMIT_BAL",
                "AGE",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]
        elif args.dataset == "parkinsons":
            cols_to_norm = [
                "age",
                "test_time",
                "motor_UPDRS",
                "Jitter(%)",
                "Jitter(Abs)",
                "Jitter:RAP",
                "Jitter:PPQ5",
                "Jitter:DDP",
                "Shimmer",
                "Shimmer(dB)",
                "Shimmer:APQ3",
                "Shimmer:APQ5",
                "Shimmer:APQ11",
                "Shimmer:DDA",
                "NHR",
                "HNR",
                "RPDE",
                "DFA",
                "PPE",
            ]

        if args.dataset == "adult":
            sensitive_attributes = ["sex"]
        elif args.dataset == "credit-card":
            sensitive_attributes = ["SEX"]
        elif args.dataset == "parkinsons":
            sensitive_attributes = ["sex"]

        if args.dataset == "adult":
            output_col_name = ">50K"
        elif args.dataset == "credit-card":
            output_col_name = "default payment next month"
        elif args.dataset == "parkinsons":
            output_col_name = "total_UPDRS"

        args.path = path
        args.num_inp_attr = num_inp_attr
        args.cols_to_norm = cols_to_norm
        args.sensitive_attributes = sensitive_attributes
        args.output_col_name = output_col_name

    return args
