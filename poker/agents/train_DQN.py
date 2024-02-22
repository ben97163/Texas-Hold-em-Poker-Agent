from game.players import BasePokerPlayer
from game.engine.hand_evaluator import HandEvaluator
from game.engine.card import Card
from game.engine.deck import Deck
from game.engine.action_checker import ActionChecker
from game.engine.round_manager import RoundManager
from game.engine.message_builder import MessageBuilder
from game.engine.poker_constants import PokerConstants
from game.engine.data_encoder import DataEncoder
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
update_freq = 50 # how often to update model
y = 0.99 # discount
start_E = 1 # starting chance of random action
end_E = 0.2 # final chance of random action
annealings_steps = 10000 # how many steps to reduce start_E to end_E
num_episodes = 500
pre_train_steps = 500 # how many steps of random action before training begin
load_model = False
path = 'final_cache/models/'
h_size = 128 # the size of final conv layer before spliting it into advantage and value streams
tau = 0.01 # rate to update target network toward primary network

suits = list(Card.SUIT_MAP.keys())
ranks = list(Card.RANK_MAP.keys())
tf.compat.v1.disable_eager_execution()

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

def get_action_by_num(action_num, valid_actions, is_train=True):
    if action_num == 0:
        action, amount = valid_actions[0]['action'], valid_actions[0]['amount']
    elif action_num == 1:
        action, amount = valid_actions[1]['action'], valid_actions[1]['amount']
    elif action_num == 2:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['min']
    elif action_num == 3:
        action, amount = valid_actions[2]['action'], valid_actions[2]['amount']['max']
    elif action_num == 4:
        action, amount = valid_actions[2]['action'], int(valid_actions[2]['amount']['max'] // 2)
        
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
    DQN Player, bot wich use Double-Dueling-DQN architecture.

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
        
    def __init__(self, h_size=128, lr=0.0001, gradient_clip_norm=500, total_num_actions=5, is_double=False,
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
        
        with open('final_project/hole_card_estimation.pkl', 'rb') as f:
            self.hole_card_est = pickle.load(f)
        
        if not is_train:
            tf.compat.v1.reset_default_graph()

        # scaler inputs
        self.scalar_input = keras.Input(shape=(None, 17 * 17 * 1), dtype=tf.float32)
        self.img_in = tf.reshape(self.scalar_input, [-1, 17, 17, 1])
        self.conv1 = layers.Conv2D(32, 5, 2, activation=tf.nn.elu)(self.img_in)
        self.conv2 = layers.Conv2D(64, 3, activation=tf.nn.elu)(self.conv1)
        self.conv3 = layers.Conv2D(self.h_size, 5, activation=tf.nn.elu)(self.conv2)
        self.conv3_flat = layers.Flatten(self.conv3)
        self.conv3_flat = layers.Dropout(self.conv3_flat)

        # feature inputs
        self.features_input = keras.Input(shape=(None, 20), dtype=tf.float32)
        self.d1 = layers.Dense(64, activation=tf.nn.elu)(self.features_input)
        self.d1 = layers.Dropout(self.d1)
        self.d2 = layers.Dense(128, activation=tf.nn.elu)(self.d1)
        self.d2 = layers.Dropout(self.d2)
        
        self.merge = tf.concat([self.conv3_flat, self.d2], axis=1)
        self.d3 = layers.Dense(256, activation=tf.nn.elu)(self.merge)
        self.d3 = layers.Dropout(self.d3)
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
            if is_train:
                self._print(len(variables))
                self._print(variables)
            self.gradients = tf.gradients(self.loss, variables)
#             self.grad_norms = tf.global_norm(self.gradients)
            self.var_norms = tf.compat.v1.global_norm(variables)
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, gradient_clip_norm)
            self.grad_norms = tf.compat.v1.global_norm(grads)
            self.trainer = tf.compat.v1.train.AdamOptimizer(lr)
#             self.update_model = self.trainer.minimize(self.loss)
            self.update_model = self.trainer.apply_gradients(zip(grads, variables))

            self.summary_writer = tf.compat.v1.summary.FileWriter('final_cache/log/DQN/')
            
        if not is_train:
            self.init = tf.compat.v1.global_variables_initializer()
            self.sess = tf.compat.v1.Session()
            self.sess.run(self.init)
        
        if is_restore:
            self.saver = tf.compat.v1.train.Saver()
            ckpt = tf.compat.v1.train.get_checkpoint_state('final_cache/models/DQN/')
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
    def _print(self, *msg):
        if self.debug:
            print(msg)
        
    def declare_action(self, valid_actions, hole_card, round_state):
        street = round_state['street']  
        bank = round_state['pot']['main']['amount']  # money in the pool
        stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]  # my current stack
        other_stacks = [s['stack'] for s in round_state['seats'] if s['uuid'] != self.uuid]  # opponents' current stack
        dealer_btn = round_state['dealer_btn']
        small_blind_pos = round_state['small_blind_pos'] # position of small blind
        big_blind_pos = round_state['big_blind_pos'] # position of big blind
        next_player = round_state['next_player'] 
        round_count = round_state['round_count']
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])] # estimate win rate

        self.features = get_street(street) # one hot encoding street state
        self.features.extend([bank, stack, dealer_btn, small_blind_pos, big_blind_pos, next_player, round_count])
        self.features.extend(other_stacks)
        self.features.append(estimation) # len 13 array
        
        img_state = img_from_state(hole_card, round_state)
        img_state = process_img(img_state)  # shape(17,17,1)
        action_num = self.sess.run(self.predict, feed_dict={self.scalar_input: [img_state],
                                                            self.features_input: [self.features]})[0]
        qs = self.sess.run(self.Q_out, feed_dict={self.scalar_input: [img_state],
                                                  self.features_input: [self.features]})[0]
        self._print(qs)
        action, amount = get_action_by_num(action_num, valid_actions)                    
        return action, amount
        
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        self._print(['Hole:', hole_card])        
        self.start_stack = [s['stack'] for s in seats if s['uuid'] == self.uuid][0]
        self._print(['Start stack:', self.start_stack])
        estimation = self.hole_card_est[(hole_card[0], hole_card[1])]
        self._print(['Estimation:', estimation])
    
    def receive_street_start_message(self, street, round_state):
        pass
            
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        end_stack = [s['stack'] for s in round_state['seats'] if s['uuid'] == self.uuid][0]
        self._print(['End stack:', end_stack])

def setup_ai():
    return DQNPlayer()