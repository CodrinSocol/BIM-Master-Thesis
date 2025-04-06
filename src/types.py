class PriceLevel:

    def __init__(self, ask_price, ask_quantity, bid_price, bid_quantity):
        self.ask_price = ask_price
        self.ask_quantity = ask_quantity
        self.bid_price = bid_price
        self.bid_quantity = bid_quantity

    def get_bid_price(self):
        return self.bid_price

    def get_bid_quantity(self):
        return self.bid_quantity

    def get_ask_price(self):
        return self.ask_price

    def get_ask_quantity(self):
        return self.ask_quantity

    def __str__(self):
        return f"PriceLevel(ask_price={self.ask_price}, ask_quantity={self.ask_quantity}, bid_price={self.bid_price}, bid_quantity={self.bid_quantity})"

class LOBSnapshot:

    def __init__(self, exchange: str, symbol: str, timestamp: int, local_timestamp: int, price_levels: list[PriceLevel]):
        self.exchange = exchange
        self.symbol = symbol
        self.timestamp = timestamp
        self.local_timestamp = local_timestamp
        self.price_levels = price_levels

    def get_exchange(self):
        return self.exchange

    def get_symbol(self):
        return self.symbol

    def get_timestamp(self):
        return self.timestamp

    def get_local_timestamp(self):
        return self.local_timestamp

    def get_all_price_levels(self):
        return self.price_levels

    def get_price_level_at_idx(self, idx: int):
        if idx < 0 or idx >= len(self.price_levels):
            raise IndexError("Index out of bounds")

        return self.price_levels[idx]

    def __str__(self):
        price_levels_str = "\n".join([str(level) for level in self.price_levels])
        return f"LOBSnapshot(exchange={self.exchange}, symbol={self.symbol}, timestamp={self.timestamp}, local_timestamp={self.local_timestamp}, price_levels=[\n{price_levels_str}\n])"