import os
import sys


def read_input_gene_list(input_fl_path):
    """
    Reads a gene list from the input file.

    Args:
        input_fl_path (str): The path to the input file.

    Returns:
        list: A list of genes read from the file.
    """
    gene_fl = open(input_fl_path, "r")
    lst_genes = gene_fl.read().split("\n")
    gene_fl.close()
    return lst_genes


api_url = "https://jsonplaceholder.typicode.com/todos/1"
