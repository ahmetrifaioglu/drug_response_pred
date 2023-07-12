import utils

lst_genes = ["PLA2G2A", "NRAS", "BUB1", "BUB1", "CTNNB1", "PIK3CA", "FGFR3", "TLR2", "APC", "MCC", "PTPN12", "BRAF", "DLC1", "RAD54B", "PTPRJ", "CCND1", "MLH3"]

for gene in lst_genes:
    print("Gene:", gene)
    utils.get_IC50_values_given_target(gene)
    