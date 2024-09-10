import os
import time
import json
import pickle
import pandas as pd
import requests
import argparse
from bs4 import BeautifulSoup
import html2text
import mygene

mg = mygene.MyGeneInfo()
parts_to_remove = [
    "##  Summary\n",
    "NEW",
    'Try the newGene table',
    'Try the newTranscript table',
    '**',
    "\nGo to the top of the page Help\n"
]

subtitles = [
    "Official Symbol",
    "Official Full Name",
    "Primary source",
    "See related",
    "Gene type",
    "RefSeq status",
    "Organism",
    "Lineage",
    "Also known as",
    "Summary",
    "Annotation information",
    "Expression",
    "Orthologs",
]

def rough_text_from_gene_name(gene_number):
    # get url
    url = f"https://www.ncbi.nlm.nih.gov/gene/{gene_number}"
    # Send a GET request to the URL
    summary_text = ''
    subtitle_dict = {}
    soup = None
    try:
        response = requests.get(url, timeout=60)
    except requests.exceptions.Timeout:
        print(f'{gene_number} time out')
        return((summary_text,soup))
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the "summary" tab content by inspecting the page's structure
        summary_tab = soup.find('div', {'class': 'rprt-section gene-summary'})

        # Check if the "summary" tab content is found
        if summary_tab:
            # Convert the HTML to plain text using html2text
            html_to_text = html2text.HTML2Text()
            html_to_text.ignore_links = True  # Ignore hyperlinks

            # Extract the plain text from the "summary" tab
            summary_text = html_to_text.handle(str(summary_tab))
            # Remove the specified parts from the original text
            for part in parts_to_remove:
                summary_text = summary_text.replace(part, ' ')
                # Replace '\n' with a space
            summary_text = summary_text.replace('  ', '').replace('    ', '').replace('\n\n', '\n').replace('\n \n \n \n', '').replace('\n ', '\n').strip()
            for subtitle in subtitles:
                first_newline = summary_text.find('\n', summary_text.find(subtitle))
                second_newline = summary_text.find('\n', first_newline + 1)
                sub_information = ''
                if summary_text.find(subtitle) != -1:
                    sub_information = summary_text[first_newline + 1:second_newline].strip() if second_newline != -1 \
                        else summary_text[first_newline + 1:].strip()
                subtitle_dict[subtitle] = sub_information
            if "Official Symbol" in subtitle_dict.keys():
                subtitle_dict["Official Symbol"] = subtitle_dict["Official Symbol"][:subtitle_dict["Official Symbol"].find('provided')].strip()
            if "Official Full Name" in subtitle_dict.keys():
                subtitle_dict["Official Full Name"] = subtitle_dict["Official Full Name"][:subtitle_dict["Official Full Name"].find('provided')].strip()
            if "Summary" in subtitle_dict.keys():
                subtitle_dict["Summary"] = subtitle_dict["Summary"][:subtitle_dict["Summary"].find('[provided')].strip()
            if "Expression" in subtitle_dict.keys():
                subtitle_dict["Expression"] = subtitle_dict["Expression"].replace(' See more', '.')
            if "Orthologs" in subtitle_dict.keys():
                subtitle_dict["Orthologs"] = subtitle_dict["Orthologs"].replace(' all', '')

        else:
            print("Summary tab not found on the page.")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return(subtitle_dict)

def main(args):
    id_to_nvbi_dict_dir = os.path.join(args.cahce_dir, "id_to_nvbi_dict.json")
    # Check if pickle exists
    if os.path.exists(id_to_nvbi_dict_dir):
        with open(id_to_nvbi_dict_dir) as file:
            gene_dict_results = json.load(file)
    else:
        # load genes used in dataset
        token_dict = pd.read_csv(args.token_dict_dir)
        if args.species_choice == "human":
            ensembl_prefix = "ENSG"
        elif args.species_choice == "mouse":
            ensembl_prefix = "ENSMUSG"
        else:
            raise
        token_dict = token_dict.loc[token_dict["ensembl_id"].str.startswith(ensembl_prefix)]
        gene_name_id_dict = dict(zip(token_dict["gene_name"], token_dict["token_id"]))
        gene_name_list = list(gene_name_id_dict.keys())
        token_id_list = []
        for name in sorted(gene_name_list):
            token_id_list.append(gene_name_id_dict[name])
        # example query to convert gene IDs into page ids for NCBI
        gene_list_results = mg.querymany(sorted(gene_name_list), scopes='symbol', species=args.species_choice)
        gene_dict_results = {token_id_list[i]: gene_list_results[i] for i in range(len(token_id_list))}
        with open(id_to_nvbi_dict_dir, 'w') as file:
            json.dump(gene_dict_results, file)

    gene_id_to_ncbi = {} # Token id to tuple like (name, ncbi_id)
    for id in gene_dict_results.keys():
        if "_id" in gene_dict_results[id].keys() and "query" in gene_dict_results[id].keys():
            gene_id_to_ncbi[id] = (gene_dict_results[id]["query"], gene_dict_results[id]["_id"])

    if not os.path.exists(args.output_file):
        with open(args.output_file, 'w') as f:
            empty = {}
            json.dump(empty, f)
    # There are many genes may not be found through mg
    not_found_list = list(gene_dict_results.keys())
    for id in gene_id_to_ncbi.keys():
        not_found_list.remove(id)

    with open(args.output_file, 'r') as f:
        previous_ = json.load(f)
    for id in not_found_list:
        previous_[id] = {}
    with open(args.output_file, 'w') as file:
        json.dump(previous_, file)

    for id in gene_id_to_ncbi.keys():
        with open(args.output_file, 'r') as f:
            previous = json.load(f)
        if id not in previous.keys() or previous[id] == {}:
            print('gene_name', gene_id_to_ncbi[id][0])
            information_dict = rough_text_from_gene_name(gene_id_to_ncbi[id][1])
            previous[id] = information_dict
            with open(args.output_file, 'w') as file:
                json.dump(previous, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_dict_dir", default="../src/token_dictionary.csv", type=str)
    parser.add_argument("--species_choice", default=None, type=str, required=True,)
    parser.add_argument("--output_file", default=None, type=str, required=True,)
    parser.add_argument("--cahce_dir", default="./", type=str)
    args = parser.parse_args()
    main(args)