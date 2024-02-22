from game.game import setup_config, start_poker

from game.engine.dealer import Dealer



class Emulator(Dealer):
    def __init__(self):
        self.game_rule = {}
        self.blind_structure = {}
        self.players_holder = {}
