import os
import time
import json
import pandas as pd
import asyncio
import argparse
from openai import AsyncOpenAI

async def request(message):
    try:
        respose = await client.chat.completions.create(messages=message,
                                                        model=model,
                                                        timeout=None)
        return respose

    except Exception as ex:
        if str(ex).startswith("Error code: 400"):
            print(ex)
        return None

async def async_generate(id, name, total, index):

    message = [{"role": "user", "content": f"give me a brief summary of mouse gene {name}"}]
    print(name)
    # Generate discription
    response = await request(message)

    if response is None:
        retry = 0
        while response is None and retry < 10:
            ind = retry + 1
            print(f"Connection failue, now retrying {ind}")
            response = await request(message)
            retry += 1
    if response is None:
        return (id, {})

    discription_tuple = (id, {"Official Symbol": name, "Official Full Name": "", "Primary source": "", "See related": "",  "Gene type": "", "RefSeq status": "", "Organism": "Mus musculus", "Lineage": "", "Also known as": "",
                              "gpt_Summary": response.choices[0].message.content.strip(),
                              "Annotation information": "", "Expression": ""})

    print(f"Finished dscription complementation of gene {name}, id_{id}, {index} of total {total}")
    return discription_tuple

async def generate_concurrently(samples, discription_file, chunk_size):
    tasks = []
    total = len(samples.keys())
    index = 0
    results = []
    print(f"Total {total}")

    try:
        for id in samples.keys():
            index += 1
            tasks.append(async_generate(id, samples[id], total, index))
        if chunk_size is not None: # not finished...
            for chunk_id in range(0, total, chunk_size):
                print(f"Starting sublist of {chunk_size}...")
                if chunk_id + chunk_size > total:
                    sublist = tasks[chunk_id:]
                else:
                    sublist = tasks[chunk_id: chunk_id + chunk_size]
                time1 = time.time()
                results += await asyncio.gather(*sublist)
                time2 = time.time()
                if int(time2 - time1) < 60:
                    time.sleep(60- int(time2 - time1))
            for result in results:
                discription_file[result[0]].update(result[1])
        else:
            print("Gathering...")

            results = await asyncio.gather(*tasks)
            for result in results:
                discription_file[result[0]].update(result[1])

        with open(args.output_file, "w") as f:
            json.dump(discription_file, f)
        print("All request finished, saved")

    except:
        print("Request interrupted, now saving...")
        with open(args.output_file, "w") as f:
            json.dump(discription_file, f)
        print("Saved")


def main_asyn(args):
    # Start from the check points
    with open(args.output_file, 'r') as file:
        complementation = json.load(file)  # {<token_id>: {name:xx, embedding:xxx}}
    print("check point loaded...")

    all_samples = {}
    test_samples = {}
    for token_id in complementation.keys():
        if "error" in complementation[token_id].keys() and ("gpt_Summary" not in complementation[token_id].keys() or complementation[token_id].get('gpt_Summary', {}) == {}):
            all_samples[token_id] = gene_id2name[token_id]
    for key in list(all_samples.keys())[:10]:
        test_samples[key] = all_samples[key]
    asyncio.run(generate_concurrently(all_samples, complementation, chunk_size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_dict_dir", default="../src/token_dictionary.csv", type=str)
    parser.add_argument("--species_choice", default=None, type=str, required=True,)
    parser.add_argument("--api_key", default=None, type=str, required=True,)
    parser.add_argument("--output_file", default=None, type=str, required=True,)
    parser.add_argument("--cahce_dir", default="./", type=str)
    args = parser.parse_args()

    token_dict = pd.read_csv(args.token_dict_dir)
    if args.species_choice == "human":
            ensembl_prefix = "ENSG"
    elif args.species_choice == "mouse":
        ensembl_prefix = "ENSMUSG"
    else:
        raise
    token_dict = token_dict.loc[token_dict["ensembl_id"].str.startswith(ensembl_prefix)]
    gene_id2name = dict(zip(token_dict["token_id"], token_dict["gene_name"]))

    client = AsyncOpenAI(api_key=args.api_key, timeout=None)
    chunk_size = 30
    model = 'gpt-3.5-turbo'
    main_asyn(args)

