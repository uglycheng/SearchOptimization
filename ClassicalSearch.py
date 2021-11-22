class A_star:
    def __init__(self,maze):
        self.maze = maze
        self.actions = [[1,0],[-1,0],[0,1],[0,-1]]
        self.points = len(maze)*len(maze[0])-sum([sum(i) for i in maze])
        self.x_max = len(maze)-1
        self.y_max = len(maze[0])-1    
            
    def search(self,x0,target,h):
        frontier_dic = {x0:0}
        explored = set()
        last_step = {x0:-1}
        cost_dic = {x0:0}
        while len(explored) < self.points:
            node = self.min_node(frontier_dic)
            node_f = frontier_dic.pop(node)
            cost_f = cost_dic.pop(node)
            if node == target:
                path = []
                while last_step[node] != -1:
                    path.append(node)
                    node = last_step[node]
                path.reverse()
                return cost_f, [x0]+path
            else:
                explored.add(node)
            for a in self.actions:
                newx,newy = node[0]+a[0],node[1]+a[1]
                if not(0<=newx<=self.x_max and 0<=newy<=self.y_max) or ((newx,newy) in explored) or self.maze[newx][newy]:
                    continue
                f = cost_f+1+h(target,(newx,newy))
                if (((newx,newy) in frontier_dic)  and (f<frontier_dic[(newx,newy)])) or ((newx,newy) not in frontier_dic):
                    frontier_dic[(newx,newy)] = f
                    cost_dic[(newx,newy)] = cost_f+1
                    last_step[(newx,newy)] = node
        return -1,[]        
            
    def min_node(self,d):
        min_key = ''
        min_val = float('inf')
        for key,value in d.items():
            if value < min_val:
                min_val = value
                min_key = key
        return min_key


if __name__ == "__main__":
    astar = A_star([
        [0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])
    c,p = astar.search((0,2),(0,0),lambda a,b:((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5)
    print(c)
    print(p)