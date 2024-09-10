import os
import time
import json
import pickle
import argparse
from openai import OpenAI

def request(message):
    try:
        respose = client.embeddings.create(input=message,
                                            model=model,
                                            timeout=None)
        return respose
    except Exception as ex:
        if str(ex).startswith("Error code: 400"):
            return None

def generate(id, summary, total, index):
    m1 = summary["Official Symbol"]
    m2 = summary["Official Full Name"]
    m3 = summary["Primary source"]
    m4 = summary["See related"]
    m5 = summary["Gene type"]
    m6 = summary["Organism"]
    m7 = summary["Lineage"]
    m8 = summary["Also known as"]
    if "error" in summary.keys():
        m9 = summary["gpt_Summary"]
    else:
        m9 = summary["Summary"]
    m10 = summary["Annotation information"]
    m11 = summary["Expression"]
    message = f"Official Symbol: {m1}\nOfficial Full Name: {m2}\nPrimary source: {m3}\nSee related: {m4}\nGene type: {m5}\nOrganism: {m6}\nLineage: {m7}\nAlso known as: {m8}\nSummary: {m9}\nAnnotation information: {m10}\nExpression: {m11}"

    # Generate embedding
    gene_embedding = request(message)

    if gene_embedding is None:
        retry = 0
        while gene_embedding is None and retry < 15:
            index = retry + 1
            print(f"Connection failue, now retrying {index}")
            gene_embedding = request(message)
            retry += 1
    if gene_embedding is None:
        raise ConnectionError("Request unsucesessful after 15 retries...")

    if "error" in summary.keys():
        embedding_tuple = (id, {"error": summary["error"], "gene_name": summary["Official Symbol"], "species": summary["Organism"], "gene_type": summary["Gene type"], "gpt_embedding": gene_embedding.data[0].embedding})
    else:
        embedding_tuple = (id, {"gene_name": summary["Official Symbol"], "embedding": gene_embedding.data[0].embedding})
    print(f"Finished embedding id_{id}, {index} of total {total}")
    return embedding_tuple

def generate_concurrently(args, samples, embedding_dict):
    tasks = []
    total = len(samples.keys())
    index = 0

    try:
        for id in samples.keys():
            time1 = time.time()
            index += 1
            try:
                result = generate(id, samples[id], total, index)
            except ConnectionError:
                continue
            embedding_dict[result[0]] = result[1]
            time2 = time.time()
            if time2 - time1 < 2:
                time.sleep(2 - int(time2 - time1))
        print("All requests finished, now saving...")
        with open(args.output_file, "wb") as f:
            pickle.dump(embedding_dict, f)
        print("Saved")

    except KeyboardInterrupt:
        print("Manually stopped, now saving...")
        with open(args.output_file, "wb") as f:
            pickle.dump(embedding_dict, f)
        print("Saved")


def main(args):
    # Start from the check points
    if not os.path.exists(args.output_file):
        with open(args.output_file, 'wb') as file:
            empty_dict = {}
            pickle.dump(empty_dict, file)
    with open(args.output_file, 'rb') as file:
        embedding_dict = pickle.load(file)  # {<token_id>: {name:xx, embedding:xxx}}

    with open(args.discription_file, 'r') as file:
        discription_file = json.load(file)

    all_samples = {}
    test_samples = {}
    for token_id in discription_file.keys():
        if token_id not in embedding_dict.keys() and discription_file[token_id] != {}:
            all_samples[token_id] = discription_file[token_id]
    for key in list(all_samples.keys())[:100]:
        test_samples[key] = discription_file[key]
    generate_concurrently(args, all_samples, embedding_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None, type=str, required=True,)
    parser.add_argument("--discription_file", default=None, type=str, required=True,)
    parser.add_argument("--output_file", default=None, type=str, required=True,)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key, timeout=None)
    model = 'text-embedding-ada-002'

    main(args)

