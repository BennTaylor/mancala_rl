from enum import Enum

GameResponse = Enum('GameResponse', ['TURN_OVER', 'TURN_CONT', 'ILLEGAL', 'GAME_OVER'])

class Mancala:
    '''
    1D array encodes state of the mancala board.
    Indexed in 'snake' order: 
        - player 1's first well is index 0
        - player 1's store is index 6
        - player 2's first well is index 7
        - player 2's store is index 13
    wells[i] = number of marbles in well
    '''
    wells = [0] * 14

    '''
    Turn counter. Refers to that which is about to be taken. (0 is only for unitialized board). 
    '''
    turn = 0

    '''
    Flag for game end.
    '''
    over = False

    '''
    Initializes mancala board with starting marbles, begins turn counter.
    '''
    def __init__(self):
        self.wells = [0] * 14
        self.wells[0:6] = [4] * 6
        self.wells[7:13] = [4] * 6
        self.turn = 1
    
    '''
    Checks if game in completed state. If it is:
        - updates boolean flag 'end'
        - cascades remaining marbles to appropriate store
    '''
    def check_end(self):
        if all([m == 0 for m in self.wells[0:6]]):
            self.wells[13] += sum(self.wells[7:13])
            self.wells[7:13] = [0] * 6
            self.over = True
            return True
        if all([m == 0 for m in self.wells[7:13]]):
            self.wells[6] += sum(self.wells[0:6])
            self.wells[0:6] = [0] * 6
            self.over = True
            return True
        return False

    '''
    Player takes action by displacing marbles from well i (counted 1-6 from player's perspective).
    - flag zero_ind indicates if input should be interpreted as zero indexed well

    Adjusts turn counter once player's turn is completed.
        - game loop should check this, prompting agent to take additional action when turn not over.

    Returns observation to be used by Agent class(es):
        - GameResponse enumeration
        - boolean flag if player 1's turn
        - array of marbles in wells         **** consider packaging this information differently
    '''
    def action(self, well, zero_ind=False):
        if self.over:
            print('Illegal action: game ended')
            return 4, None, None
        well += zero_ind
        # boolean flag for if player 1's turn
        p1 = (self.turn % 2 == 1)

        if well < 1 or well > 6:
            print('Illegal action: out of bounds well')
            return 3, p1, self.wells

        # convert to index as stored by class
        if p1:
            w_ind = well - 1
        else:
            w_ind = well + 6

        # if empty well: action illegal. return
        if self.wells[w_ind] == 0:
            print(f'Illegal action: well {well} already empty')
            return 3, p1, self.wells
        
        # displace marbles
        marbles = self.wells[w_ind]
        self.wells[w_ind] = 0
        i = w_ind + 1
        while marbles > 0:
            # skip opponent's store
            if p1 and i == 13:
                i = 0       # increment then mod 14
            elif not p1 and i == 6:
                i += 1
            self.wells[i] += 1
            i = (i + 1) % 14
            marbles -= 1
            
        # capture opposite marbles when landing in empty well
        if p1 and i >= 1 and i < 7 and self.wells[i - 1] == 1 and self.wells[13 - i] != 0:
            self.wells[6] += self.wells[13 - i] + 1
            self.wells[i - 1] = 0
            self.wells[13 - i] = 0
        if not p1 and i >= 8 and i < 14 and self.wells[i - 1] == 1 and self.wells[13 - i] != 0:
            self.wells[13] += self.wells[13 - i] + 1
            self.wells[i - 1] = 0
            self.wells[13 - i] = 0

        if self.check_end():
            if self.wells[6] + self.wells[13] != 48:
                print(f'MANCALA ERROR: impossible sum reached; board: {self.wells}')
            return 4, True, self.wells
        
        # account for turn continuation when last marble put in well
        if p1 and i != 7:
            self.turn += 1
            return 1, False, self.wells
        elif not p1 and i != 0:
            self.turn += 1
            return 1, True, self.wells
        
        return 2, p1, self.wells

    '''
    Printing functionality for terminal play.
    *** EITHER make this a to_string function that sends a string of the intended terminal output 
        OR print directly from here ***
    '''
    def print_board(self):
        print(f'      1   2   3   4   5   6')
        print(f'+ - - - - - - - - - - - - - - - +')
        print(f'|   | {self.wells[0]} | {self.wells[1]} | {self.wells[2]} | {self.wells[3]} | {self.wells[4]} | {self.wells[5]} |   |')
        print(f'| {self.wells[13]} |                       | {self.wells[6]} |')
        print(f'|   | {self.wells[12]} | {self.wells[11]} | {self.wells[10]} | {self.wells[9]} | {self.wells[8]} | {self.wells[7]} |   |')
        print(f'+ - - - - - - - - - - - - - - - +')
        print(f'      6   5   4   3   2   1')

    '''
    Observation function to be used by Agent class(es).
    Returns:
    - list of wells (reindixed as if current player always player 1)
    - list of legal moves (0-indexing on player's 6 wells)
    '''
    def observation(self):
        if self.turn % 2 == 1:
            return self.wells, [i for i,m in enumerate(self.wells[0:6]) if m != 0]
        else:
            wells = self.wells[7:14] + self.wells[0:7]
            return wells, [i for i,m in enumerate(wells[0:6]) if m != 0]
