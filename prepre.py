# coding: utf-8

from tqdm import tqdm
import json

from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-n", "--name", default="reviews_Musical_Instruments_5")
parser.add_argument("-d", "--directory", default="data")
args = parser.parse_args()

path = '{}/{}.json'.format(args.directory, args.name)

file = open(path)
     
with open('data/{}_ratings.csv'.format(args.name), 'wt') as out_file:
    
    users_count = 0
    items_count = 0
    
    user_to_id = {}
    item_to_id = {}
    
    out_file.write('userId,movieId,rating\n')
    
    for rating_id, line in tqdm(enumerate(file), desc="Reading data from file"):

        line = line.split('\t')[-1]
        line = line[line.index('{'):]

        sample = json.loads(line)

        current_user_id = user_to_id.get(sample['reviewerID'], users_count)
        if current_user_id == users_count:
            user_to_id[sample['reviewerID']] = users_count
            users_count += 1
            
        current_item_id = item_to_id.get(sample['asin'], items_count)
        if current_item_id == items_count:
            item_to_id[sample['asin']] = items_count
            items_count += 1
        
        if int(sample['overall']) < 3.5:
            continue

        out_file.write("{},{},{:1}\n".format(current_user_id, current_item_id, int(sample['overall'])))


with open("data/{}_item_mapping.json".format(args.name), "wt", encoding="utf-8") as f_out:
    json.dump(fp=f_out, obj=item_to_id)




