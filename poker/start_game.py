import json
from game.game import setup_config, start_poker
from agents.call_player import setup_ai as call_ai
from agents.random_player import setup_ai as random_ai
from agents.console_player import setup_ai as console_ai
from agents.monte_carlo import setup_ai as MTPlayer
from agents.DQN import setup_ai as my_ai
from agents.DQN import DQNPlayer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#from baseline5 import setup_ai as baseline5_ai

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="You", algorithm=console_ai())
config.register_player(name="Benjamin", algorithm=DQNPlayer(is_restore=True, is_train=False))

game_result = start_poker(config, verbose=1)
print(json.dumps(game_result, indent=4))
exit()

config = setup_config(max_round=20, initial_stack=1000, small_blind_amount=5)
config.register_player(name="you", algorithm=console_ai())
config.register_player(name="b09703028", algorithm=DQNPlayer(is_restore=True, is_train=False))
win = 0
for i in range(10):
    game_result = start_poker(config, verbose=1)
    #print(json.dumps(game_result, indent=4))
    if game_result['players'][0]['stack'] <= game_result['players'][1]['stack']:
        win += 1
        print(f"win at time {i}")

print(win/10)

## Play in interactive mode if uncomment
# config.register_player(name="me", algorithm=console_ai())
#game_result = start_poker(config, verbose=1)


#print(json.dumps(game_result, indent=4))
