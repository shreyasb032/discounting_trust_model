class ConstantDF:
    """
    A class that gives a constant discount factor irrespective of the search site number
    """

    def __init__(self, discount_factor: float):
        """
        :param discount_factor: The constant discount factor
        """
        super().__init__()
        self.name = "Constant"
        self.df = discount_factor

    def get_value(self, site_number: int):
        """
        Returns the constant discount factor
        :param site_number: the current search site
        :return: the constant discount factor
        """
        return self.df
