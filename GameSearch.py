import copy

class Game:
    # Here is the game https://leetcode-cn.com/problems/stone-game/
    def __init__(self,stones):
        self.stones = stones
        self.A = 0
        self.B = 0
        self.head = 0
        self.tail = len(stones)-1
        self.moves = []
        
    def step(self,player,head_tail):
        if self.head>self.tail:
            return 1
            print('Game Already Over')
        if player == 'A':
            if head_tail == 'h':
                self.A += self.stones[self.head]
                self.head += 1
            elif head_tail == 't':
                self.A += self.stones[self.tail]
                self.tail -= 1
            else:
                raise ValueError('head_tail must be t or h')
        elif player == 'B':
            if head_tail == 'h':
                self.B += self.stones[self.head]
                self.head += 1
            elif head_tail == 't':
                self.B += self.stones[self.tail]
                self.tail -= 1
            else:
                raise ValueError('head_tail must be t or h')
        else:
            raise ValueError('player must be A or B')
            
        self.moves.append((player,head_tail))
        return 0
        
    def undo(self):
        player,head_tail = self.moves.pop()
        if player == 'A':
            if head_tail == 'h':
                self.head -= 1
                self.A -= self.stones[self.head]
            elif head_tail == 't':
                self.tail += 1
                self.A -= self.stones[self.tail]
        elif player == 'B':
            if head_tail == 'h':
                self.head -= 1
                self.B -= self.stones[self.head]
            elif head_tail == 't':
                self.tail += 1
                self.B -= self.stones[self.tail]



class Node:
    def __init__(self,score,player):
        self.children = []
        self.score = score
        if player not in set(['self','opponent']):
            raise ValueError('player must be self or opponent')
        self.player = player
        


class Minimax:
    
    def __init__(self,layer,game,player):
        self.simulator = copy.deepcopy(game)
        score = self.simulator.A - self.simulator.B
        if player == 'B':
            score = -score
        self.player = player
        self.node = Node(score,'self')
        self.layer = layer
        self.construct_tree(self.node,0)
        
        
    def construct_tree(self,node,cur_layer):
        if cur_layer == self.layer:
            return
        if self.simulator.head>self.simulator.tail:
            return
        if len(node.children)==0 and node.player == 'self':
            for move in ['h','t']:
                self.simulator.step(self.player,move)
                score = self.simulator.A - self.simulator.B
#                 print(self.simulator.A,self.simulator.B,score,self.simulator.head,self.simulator.tail)
                self.simulator.undo()
                if self.player == 'B':
                    score = -score
                node.children.append((move,Node(score,'opponent')))
        elif len(node.children)==0 and node.player == 'opponent':
            for move in ['h','t']:
                self.simulator.step((set(['A','B'])-set([self.player])).pop(),move)
                score = self.simulator.A - self.simulator.B
#                 print(self.simulator.A,self.simulator.B,score,self.simulator.head,self.simulator.tail)
                self.simulator.undo()
                if self.player == 'B':
                    score = -score
                node.children.append((move,Node(score,'self')))
        
        if node.player == 'self':
            player = self.player
        else:
            player = (set(['A','B'])-set([self.player])).pop()
        for child in node.children:
            self.simulator.step(player,child[0])
            self.construct_tree(child[1],cur_layer+1)
            self.simulator.undo()
            
    def step(self,move):
        for child in self.node.children:
            if child[0] == move:
                self.simulator.step(self.player,child[0])
                self.node = child[1]
                self.construct_tree(self.node,0)
                
    def minimax(self,node):
        if node.children == []:
            return None,node.score
        if node.player == 'self':
            max_move = ''
            max_score = -float('inf')
            for c in node.children:
                s = self.minimax(c[1])[1]
#                 s = c[1].score
                if s > max_score:
                    max_score = s
                    max_move = c[0]
            return max_move,max_score
        if node.player == 'opponent':
            min_move = ''
            min_score = float('inf')
            for c in node.children:
                s = self.minimax(c[1])[1]
#                 s = c[1].score
                if s < min_score:
                    min_score = s
                    min_move = c[0]
            return min_move,min_score


if __name__ == '__main__':
    game = Game([2,7,3,5,4,8,2,9])
    aiA = Minimax(8,game,'A')
    move,s_ = aiA.minimax(aiA.node)
    print('Minimax Answer')
    print('A',move,s_)
   

   #This game can be solved using dynamic programming
    def stoneGame(piles):
        dp = piles.copy()
        for i in range(1,len(piles)):
            dp_temp = [0] * (len(dp)-1)
            for j in range(len(dp_temp)):
                dp_temp[j] = max(piles[j]-dp[j+1],piles[j+i]-dp[j])
            dp = dp_temp.copy()
        return dp[0]
    print('DP Answer')
    print(stoneGame([2,7,3,5,4,8,2,9]))

    