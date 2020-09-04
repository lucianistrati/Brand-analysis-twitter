class Company:

    def __init__(self, company, industry, start_date, end_date):
        """
        Class that contains relevant informations about an industry
        Parameters
        ----------
        company : string
                the name of the company
        industry : string
                the name of the industry
        start_date : string
                the first day of the analysing period
                has the format "MM-DD-YYYY"
        end_date : string
                the lasst day of the analysing period
                has the format "MM-DD-YYYY"
        """
        self.company = company
        self.industry = industry
        self.start_date = start_date
        self.end_date = end_date
        self.multi_year_scores = None
        self.yearly_averaged_scores = None
        self.ereputation_score = 0
        self.number_of_tweeets = 0
