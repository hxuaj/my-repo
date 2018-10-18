from random import randint

index = 0
num = 0
player = []
player_num = input("How many of you are going to play?:")

while num < int(player_num):
    player_name = input("Please enter player{}'s name.".format(num+1))
    num = num+1
    player.append(player_name)

while True:
    n = index%len(player)
    game_round = index // len(player) + 1
    i = input("Round{}. It's time for {} to dice! Press Enter! (Input h for more info.)\n"
              .format(game_round, player[n]))
    
    if i == 'help':
        print("Press enter to dice. Input q if you want to quit.")
    if i == 'q':
        break
    if i == '':
        index = index + 1
        result = randint(1,6)
        print("Round{} --> player:{} result:{}\n".format(game_round, player[n], result))