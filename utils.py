import os
import pandas as pd


def write_line(dict_in: dict, log_dir: str):
    # record loss in real time.
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    with open(log_dir, 'a') as file:
        for key, value in dict_in.items():
            if isinstance(key, float):
                key = round(key, 4)
            if isinstance(value, float):
                value = round(value, 6)
            file.write(str(key) + ' : ' + str(value) + '\n')


def csv_record(input_dict, csv_dir, index=None):
    dataframe = pd.DataFrame(input_dict, index=index)

    if not os.path.exists(csv_dir):  # writing mode with header
        dataframe.to_csv(csv_dir, index=index, mode='w', encoding='utf-8-sig')
    else:                            # append mode without header
        dataframe.to_csv(csv_dir, index=index, mode='a', encoding='utf-8-sig', header=False)


