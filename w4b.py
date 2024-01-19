import random as rd

# 123 = front = white white black, 456 = back = white, black, black
def pick_a_card():
    whitewhite = 0
    whiteblack = 0
    blackwhite = 0
    blackblack = 0

    # pick a card
    for i in range(5000):
        rng = rd.randint(1,6)
        # sees white
        if rng == 1 or rng == 2 or rng == 4:
            # other side could be 1,4,5
            rngAgain = rd.randint(1,3)
            if rngAgain == 1 or rngAgain == 2:
                whitewhite += 1
            else:
                whiteblack += 1
        # sees black
        if rng == 3 or rng == 5 or rng == 6:
            # other side could be 2,3,6
            rngAgain = rd.randint(1,3)
            if rngAgain == 1 or rngAgain == 2:
                blackblack += 1
            else:
                blackwhite += 1      
    return (whitewhite,whiteblack,blackblack,blackwhite)

print(pick_a_card())
