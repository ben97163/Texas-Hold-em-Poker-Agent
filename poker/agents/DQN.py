from game.players import BasePokerPlayer
from game.engine.card import Card
from game.engine.hand_evaluator import HandEvaluator
import random as rand
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
# import os

suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())
tf.compat.v1.disable_eager_execution()

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

def gen_card_im(card):
    a = np.zeros((4, 13))
    s = suits.index(card.suit)
    r = ranks.index(card.rank)
    a[s, r] = 1
    return np.pad(a, ((6, 7), (2, 2)), 'constant', constant_values=0)

streep_map = {
    'preflop': 0,
    'flop': 1,
    'turn': 2,
    'river': 3
}

def get_street(s):  # one hot encoding
    val = [0, 0, 0, 0]
    val[streep_map[s]] = 1
    return val

def process_img(img):
    return np.reshape(img, [17 * 17 * 1])

def get_action_by_num(action_num, valid_actions, threshold, stack, is_train=True):
    if action_num == 0:# fold
        action, amount = valid_actions[0]['action'], valid_actions[0]['amount']
    elif action_num == 1:# call
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    elif action_num == 2:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']
    elif action_num == 3:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']
    elif action_num == 4:
        action, amount = valid_actions[2]['action'], min(threshold - stack + 0.5, valid_actions[2]['amount']['max'])
        
    if not is_train and amount == -1:
        print(action, amount)
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    return action, amount

def img_from_state(hole_card, round_state):
    imgs = np.zeros((8, 17, 17))
    for i, c in enumerate(hole_card):
        imgs[i] = gen_card_im(Card.from_str(c))

    for i, c in enumerate(round_state['community_card']):
        imgs[i + 2] = gen_card_im(Card.from_str(c))

    imgs[7] = imgs[:7].sum(axis=0)
#     return imgs
    return np.swapaxes(imgs, 0, 2)[:, :, -1:]

class DQNPlayer(BasePokerPlayer):
    '''
    Parametrs
    ---------
    h_size : shape of layer after conv part (also before double part too)

    lr : learning rate of the optimizer

    gradient_clip_norm : gradients of the loss function will be clipped by this value
    
    total_num_actions : the number of actions witch agent can choose

    is_double : whether or not to use the double architecture

    is_main : whether or not to use this agent as main (when using the dueling architecture)

    is_restore : wheter or not to use pretrained weight of the network

    is_train : whether or not to use this agent for training

    is_debug  wheter or not to print the debug information
    '''
        
    def __init__(self, h_size=128, lr=0.0001, gradient_clip_norm=500, total_num_actions=5, e=1, is_double=False,
                 is_main=True, is_restore=False, is_train=True, debug=False):              
        self.h_size = h_size
        self.lr = lr
        self.gradient_clip_norm = gradient_clip_norm
        self.total_num_actions = total_num_actions
        self.is_double = is_double
        self.is_main = is_main
        self.is_restore = is_restore
        self.is_train = is_train
        self.debug = debug
        self.last_img_state = None
        self.last_features = None
        self.last_action_num = None
        self.fold_flag = False
        self.win_table = [0,1145,1135,1130,1120,1115,1105,1100,1090,1085,1075,1070,1060,1055,1045,1040,1030,1025,1015,1010,1000]
        self.lose_table = [0,860,865,875,880,890,895,905,910,920,925,935,940,950,955,965,970,980,985,995,1000]
        with open('src/hole_card_estimation.pkl', 'rb') as f:
            self.hole_card_est = pickle.load(f)
        
        if not is_train:
            tf.compat.v1.reset_default_graph()

        # scaler inputs
        #self.scalar_input = keras.Input(shape=(17 * 17 * 1), dtype=tf.float32)
        self.scalar_input = tf.compat.v1.placeholder(tf.float32, [None, 17 * 17 * 1])
        self.img_in = tf.reshape(self.scalar_input, [-1, 17, 17, 1])
        self.conv1 = layers.Conv2D(32, 5, 2, activation=tf.nn.elu)(self.img_in)
        self.conv2 = layers.Conv2D(64, 3, activation=tf.nn.elu)(self.conv1)
        self.conv3 = layers.Conv2D(self.h_size, 5, activation=tf.nn.elu)(self.conv2)
        self.conv3_flat = layers.Flatten()(self.conv3)
        self.conv3_flat = layers.Dropout(0.5)(self.conv3_flat)

        # feature inputs
        #self.features_input = keras.Input(shape=(13), dtype=tf.float32)
        self.features_input = tf.compat.v1.placeholder(tf.float32, [None, 13])
        self.d1 = layers.Dense(64, activation=tf.nn.elu)(self.features_input)
        self.d1 = layers.Dropout(0.5)(self.d1)
        self.d2 = layers.Dense(128, activation=tf.nn.elu)(self.d1)
        self.d2 = layers.Dropout(0.5)(self.d2)
        
        self.merge = tf.concat([self.conv3_flat, self.d2], axis=1)
        self.d3 = layers.Dense(256, activation=tf.nn.elu)(self.merge)
        self.d3 = layers.Dropout(0.5)(self.d3)
        self.d4 = layers.Dense(self.h_size, activation=tf.nn.elu)(self.d3)
        self.Q_out = layers.Dense(5)(self.d4)
            
        self.predict = tf.argmax(self.Q_out, 1)
        
        self.target_Q = tf.compat.v1.placeholder(tf.float32, [None])
        self.actions = tf.compat.v1.placeholder(tf.int32, [None])
        self.actions_onehot = tf.one_hot(self.actions, total_num_actions, dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.Q - self.target_Q)
        self.loss = tf.reduce_mean(self.td_error)
        
        if is_main:
            variables = tf.compat.v1.trainable_variables() # [:len(tf.trainable_variables()) // 2]
            #if is_train:
                #self._print(len(variables))
                #self._print(variables)
            self.gradients = tf.gradients(self.loss, variables)
#             self.grad_norms = tf.global_norm(self.gradients)
            self.var_norms = tf.compat.v1.global_norm(variables)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, gradient_clip_norm)
            self.grad_norms = tf.compat.v1.global_norm(grads)
            self.trainer = tf.compat.v1.train.AdamOptimizer(lr)
#             self.update_model = self.trainer.minimize(self.loss)
            self.update_model = self.trainer.apply_gradients(zip(grads, variables))

#            self.summary_writer = tf.compat.v1.summary.FileWriter('final_cache/log/DQN/')
            
        if not is_train:
            self.init = tf.compat.v1.global_variables_initializer()
            self.sess = tf.compat.v1.Session()
            self.sess.run(self.init)
        
        if is_restore:
            self.saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state('src/newModels/')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
    def declare_action(self, valid_actions, hole_card, round_state):
        fold_action_info = valid_actions[0]
        call_action_info = valid_actions[1]
        raise_action_info = valid_actions[2]

        street = round_state['street']  
        bank = round_state['pot']['main']['amount']  # money in the pool
        stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]  # my current stack
        other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid]  # opponents' current stack
        dealer_btn = round_state['dealer_btn']
        small_blind_pos = round_state['small_blind_pos'] # position of small blind
        big_blind_pos = round_state['big_blind_pos'] # position of big blind
        next_player = round_state['next_player'] 
        round_count = round_state['round_count']
        community_card = round_state['community_card']
        blind = 10 if round_state['seats'][big_blind_pos]['name'] == 'b09703028' else 5

        estimation = self.hole_card_est[(hole_card[0], hole_card[1])] # estimate win rate
        win_rate = estimate_hole_card_win_rate(nb_simulation=10000, nb_player=2,
                                               hole_card=gen_cards(hole_card),
                                               community_card=gen_cards(community_card))
#        print(win_rate)


        if stack > self.win_table[round_count]:  # already won
            action, amount = fold_action_info["action"], fold_action_info["amount"]
            return action, amount

        if self.round_bet_amount == 0:
            self.round_bet_amount = blind
        if round_count == 20:  # last round
            if 1000 - stack > blind:
                action, amount =  'raise',  max(raise_action_info['amount']['min'], min(self.win_table[round_count] - stack + 0.5, raise_action_info['amount']['max']))
                if amount == -1:
                    return 'call', call_action_info["amount"]
                else:
                    return 'raise', amount
            elif stack - 1000 < blind: 
                return 'call', call_action_info["amount"]
        if street == 'preflop' and win_rate < 0.5 and stack > self.lose_table[round_count]:
            return 'fold', 0
        
        if street == 'flop' and win_rate > 0.8 and stack < other_stacks[0]:
            action, amount = 'raise', raise_action_info['amount']['max']
            if amount == -1:
                return 'call', call_action_info["amount"]
            else:
                return action, amount

        if street == 'river' and win_rate > 0.75 and stack < other_stacks[0]:
            action, amount = 'raise', raise_action_info['amount']['max']
            if amount == -1:
                return 'call', call_action_info["amount"]
            else:
                return action, amount
        
        self.features = get_street(street) # one hot encoding street state
        self.features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
        self.features.extend(other_stacks)
        self.features.append(estimation) # len 13 array
        img_state = img_from_state(hole_card, round_state)
        self.img_state = process_img(img_state)  # shape(17,17,1)
        action_num = self.sess.run(self.predict, feed_dict={self.scalar_input: [self.img_state], self.features_input: [self.features]})[0]
        
        action, amount = get_action_by_num(action_num, valid_actions, self.win_table[round_count], stack)   
        self.last_action_num, self.last_features, self.last_img_state =  action_num, self.features.copy(), self.img_state.copy()

        if action == 'raise' and amount < call_action_info["amount"]:
            action, amount = 'call', call_action_info["amount"]
        if amount > stack and action == 'raise':
            action, amount = 'raise', stack 
        
        if action == 'raise' and street == 'preflop' and amount > 100:
            action, amount = 'call', call_action_info["amount"]

        if street == 'preflop':
            self.round_bet_amount = amount if amount != 0 else self.round_bet_amount
            self.previous_street_bet_amount = self.round_bet_amount
        else:
            self.round_bet_amount = amount + self.previous_street_bet_amount

        if (win_rate < 0.4 and stack >= self.lose_table[round_count]) or (call_action_info["amount"] != 0 and stack - call_action_info["amount"] < self.lose_table[round_count] and win_rate < 0.55) or (call_action_info["amount"] != 10 and stack - call_action_info["amount"] < self.lose_table[round_count] and win_rate < 0.55):
                action, amount = 'fold', 0
        
        if action == 'fold' and stack < self.lose_table[round_count]: # can't fold if lost
            action, amount = 'call', call_action_info["amount"]

        if amount > raise_action_info['amount']['max'] and action == 'raise':
            action, amount = 'raise', raise_action_info['amount']['max']
        
        return action, amount
        
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        #self._print(['Hole:', hole_card])        
        self.start_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]
        #self._print(['Start stack:', self.start_stack])
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])]
        self.round_bet_amount = 0
        #self._print(['Estimation:', estimation])
    
    def receive_street_start_message(self, street, round_state):
        pass
            
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        end_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        opponent_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid][0]
        stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        if (stack - opponent_stack) > (21 - round_state['round_count']) * 15:
            self.fold_flag = True
        #self._print(['End stack:', end_stack])

def setup_ai():
    return DQNPlayer()