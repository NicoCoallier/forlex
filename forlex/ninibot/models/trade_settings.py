class TradeSettings:
    def __init__(self, ob):
        self.n_ma = ob["n_ma"]
        self.n_std = ob["n_std"]
        self.maxspread = ob["maxspread"]
        self.mingain = ob["mingain"]
        self.riskreward = ob["riskreward"]

    def __repr__(self):
        return str(vars(self))

    @staticmethod
    def settings_to_str(settings):
        ret_str = "Trade Settings:\n"
        for v in settings.values():
            ret_str += f"{v}\n"

        return ret_str
