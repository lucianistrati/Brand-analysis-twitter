import numpy as np

def delete_company(company, industry):
    ind_to_comp = np.load("industry_to_companies.npy", allow_pickle=True)
    for company_name in ind_to_comp[()][industry]:
        if company == company_name:
            ind_to_comp[()][industry].remove(company)
            np.save("industry_to_companies.npy", ind_to_comp)
            return 1
    return -1

def add_company(company, industry):
    ind_to_comp = np.load("industry_to_companies.npy", allow_pickle=True)
    if delete_company(company, industry)==-1:
        ind_to_comp[()][industry].append(company)
    np.save("industry_to_companies.npy", ind_to_comp)

if __name__:
    ind_to_comp = np.load("industry_to_companies.npy", allow_pickle=True)
    print(ind_to_comp)