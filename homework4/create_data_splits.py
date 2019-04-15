import argparse
import os
import random

"""
This file splits the data into train and validation sets
""" 


def read_data(data_path):
    data = {}
    files = os.walk(data_path)
    for root, _, names in files:
        for name in names:
            lang = name.split('.')[0]
            data[lang] = []
            with open(os.path.join(root, name), encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    data[lang].append(name)
    return data


def split_data(data, val_size):
    train = {}
    val = {}
    val_size = 1 - val_size
    for k in data.keys():
        names = list(data[k])
        random.shuffle(names)
        total_len = len(names)
        train_names = names[:int(total_len * val_size)]
        val_names = names[int(total_len * val_size):]
        train[k] = train_names
        val[k] = val_names
    return train, val


def write_data(data_path, data, mode='train'):
    split_path = os.path.join(data_path, mode)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    for k in data.keys():
        with open(os.path.join(split_path, k + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(data[k]))




parser = argparse.ArgumentParser(description='Split the data into train and validation.')
parser.add_argument('data_path', type=str, help='Path to the text files with names')
parser.add_argument('--val_size', '-v', type=float, default=0.3, help='Fraction of a train set to be used as validation')

args = parser.parse_args()

names = read_data(args.data_path)
train, val = split_data(names, args.val_size)
write_data(args.data_path, train, 'train')
write_data(args.data_path, val, 'val')