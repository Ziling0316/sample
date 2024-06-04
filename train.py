from othello import OthelloGame
from othello.bots.DeepLearning import BOT

class Human:
    def getAction(self, game, color):
        print('input coordinate:', end='')
        coor=input()
        return (int(coor[1])-1, ord(coor[0])-ord('A'))
BOARD_SIZE=10


bot=BOT(board_size=BOARD_SIZE)
args={
    'num_of_generate_data_for_train': 256,
    'epochs': 50,
    'batch_size': 10,
    'verbose': True
}
bot.self_play_train(args)


#game=OthelloGame(BOARD_SIZE)
#game.play(black=bot, white=Human())

