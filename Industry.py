import numpy as np


class Industry:
    def __init__(self, industry, start_date, end_date):
        """
        Class that contains relevant informations about an industry
        Parameters
        ----------
        industry : string
                the name of the industry
        start_date : string
                the first day of the analysing period
                has the format "MM-DD-YYYY"
        end_date : string
                the lasst day of the analysing period
                has the format "MM-DD-YYYY"
        """
        self.industry = industry
        self.start_date = start_date
        self.end_date = end_date
        self.multi_year_scores = None
        self.yearly_averaged_scores = None
        self.ereputation_score = 0
        self.number_of_tweeets = 0
        self.list_of_companies = self.build_companies_list()

    def build_companies_list(self):
        ind_to_comp = np.load(
            "App/templates/industry_to_companies.npy",
            allow_pickle=True)
        companies = []
        for comp in ind_to_comp[()][self.industry]:
            companies.append(comp)
        return companies
