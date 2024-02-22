from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
import random as rand
import numpy as np
from multiprocessing import Process, Pool
import multiprocessing
import pickle
from itertools import permutations
def gen_cards(cards_str):
    return [Card.from_str(s) for s in cards_str]

def _fill_community_card(base_cards, used_card):
    need_num = 5 - len(base_cards)
    return base_cards + _pick_unused_card(need_num, used_card)

def _pick_unused_card(card_num, used_card):
    used = [card.to_id() for card in used_card]
    unused = [card_id for card_id in range(1, 53) if card_id not in used]
    choiced = rand.sample(unused, card_num)
    return [Card.from_id(card_id) for card_id in choiced]

def _montecarlo_simulation(nb_player, hole_card, community_card):
    community_card = _fill_community_card(community_card, used_card=hole_card+community_card)
    unused_cards = _pick_unused_card((nb_player-1)*2, hole_card + community_card)
    opponents_hole = [unused_cards[2*i:2*i+2] for i in range(nb_player-1)]
    opponents_score = [HandEvaluator.eval_hand(hole, community_card) for hole in opponents_hole]
    my_score = HandEvaluator.eval_hand(hole_card, community_card)
    return 1 if my_score >= max(opponents_score) else 0

def estimate_hole_card_win_rate(nb_simulation, nb_player, hole_card, community_card=None):
    if not community_card: community_card = []
    win_count = sum([_montecarlo_simulation(nb_player, hole_card, community_card) for _ in range(nb_simulation)])
    return 1.0 * win_count / nb_simulation

class MCPlayer(BasePokerPlayer): # 後手對方call我領先不fold
    def __init__(self):
        self.nb_player = 2
        self.win_table = [0,1145,1135,1130,1120,1115,1105,1100,1090,1085,1075,1070,1060,1055,1045,1040,1030,1025,1015,1010,1000]
        self.lose_table = [0,860,865,875,880,890,895,905,910,920,925,935,940,950,955,965,970,980,985,995,1000]
        self.win = False
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        call_action_info = valid_actions[1]
        fold_action_info = valid_actions[0]
        raise_action_info = valid_actions[2]
        opponent_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid][0]
        stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        community_card = round_state['community_card']
        street = round_state['street']  
        bank = round_state['pot']['main']['amount']  # money in the pool
        small_blind_pos = round_state['small_blind_pos'] # position of small blind
        big_blind_pos = round_state['big_blind_pos'] # position of big blind
        round_count = round_state['round_count']
        is_losing = opponent_stack > stack
        lost = stack < self.lose_table[round_count]
        win = stack > self.lose_table[round_count]
        if stack > self.win_table[round_count]:  # already won
            action, amount = fold_action_info["action"], fold_action_info["amount"]
            return action, amount
        
        if self.raised is True:
            return 'call', call_action_info["amount"]

        if win:
            raise_action_info['amount']['max'] = max(100, raise_action_info['amount']['max'])

        win_rate = estimate_hole_card_win_rate(nb_simulation=10000, nb_player=self.nb_player,
                                               hole_card=gen_cards(hole_card),
                                               community_card=gen_cards(community_card))
        
        if raise_action_info['amount']['min'] == -1: # all in
            return 'call', call_action_info["amount"]

        blind = 10 if round_state['seats'][big_blind_pos]['name'] == 'b09703028' else 5
        if round_count == 20 and (stack - 1000 < blind or 1000 - stack > blind):  # last round
            return 'raise', raise_action_info['amount']['max']
        

        if street == 'preflop':
            if self.round_bet_amount == 0:
                self.round_bet_amount = blind
            if (win_rate < 0.4 and stack - self.round_bet_amount >= self.lose_table[round_count] and call_action_info['amount'] != 10) or (call_action_info["amount"] != 0 and stack - self.round_bet_amount - call_action_info["amount"] < self.lose_table[round_count] and win_rate < 0.6):
                action, amount = 'fold', 0
            elif win_rate >= 0.5 and win_rate < 0.7:
                action, amount = 'call', call_action_info["amount"]
            elif win_rate > 0.8:
                action = np.random.choice(['raise', 'call'],p=[0.7, 0.3])
                if action == 'raise':
                    action, amount, self.raised = 'raise', min(raise_action_info['amount']['min'] * 3 , raise_action_info['amount']['max']), True
                else:
                    action, amount = 'call', call_action_info["amount"]
            else:
                action, amount = 'raise', raise_action_info['amount']['min'] 
                self.round_bet_amount = amount
            
            self.round_bet_amount = amount if amount != 0 else self.round_bet_amount
            self.previous_win_rate = win_rate
            return action, amount 
    
        elif street == 'flop':
            if (win_rate < 0.4 and stack - self.round_bet_amount >= self.lose_table[round_count]) or (call_action_info["amount"] != 0 and stack - self.round_bet_amount - call_action_info["amount"] < self.lose_table[round_count] and win_rate < 0.55):
                action, amount = 'fold', 0
            elif win_rate >= 0.6 and win_rate < 0.8:
                action, amount, self.raised = 'raise', min(raise_action_info['amount']['min'] * 2 , raise_action_info['amount']['max']), True
            elif win_rate > 0.8:
                action, amount, self.raised = 'raise', min(self.win_table[round_count] - stack + 0.5, raise_action_info['amount']['max']), True
            else:
                action, amount = 'call', call_action_info["amount"]

            self.round_bet_amount = amount if amount != 0 else self.round_bet_amount
            self.previous_street_bet_amount = self.round_bet_amount  
            return action, amount
        
        elif street == 'turn':
            if (win_rate < 0.4 and stack - self.round_bet_amount >= self.lose_table[round_count]) or (call_action_info["amount"] != 0 and stack - self.round_bet_amount - call_action_info["amount"] < self.lose_table[round_count] and win_rate < 0.5):
                action, amount = 'fold', 0
            elif win_rate > 0.8:
                action, amount, self.raised = 'raise', min(self.win_table[round_count] - stack + 0.5, raise_action_info['amount']['max']), True
            else:
                action, amount = 'call', call_action_info["amount"]
            
            self.round_bet_amount = amount + self.previous_street_bet_amount 
            self.previous_win_rate = win_rate   
            return action, amount
        
        elif street == 'river':
            if (win_rate < 0.5 and stack - self.round_bet_amount >= self.lose_table[round_count]) or (call_action_info["amount"] != 0 and stack - self.round_bet_amount - call_action_info["amount"] < self.lose_table[round_count] and win_rate < 0.5):
                action, amount = 'fold', 0
            elif win_rate > 0.8:
                action, amount, self.raised = 'raise', min(self.win_table[round_count] - stack + 0.5, raise_action_info['amount']['max']), True
            else:
                action, amount = 'call', call_action_info["amount"]
            
            self.round_bet_amount = amount + self.previous_street_bet_amount
            self.previous_win_rate = win_rate   
            return action, amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_bet_amount = 0
        pass

    def receive_street_start_message(self, street, round_state):
        self.raised = False
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return MCPlayer()
