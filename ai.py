import os
import matplotlib.pyplot as plt
from heapq import heappop
from heapq import heappush
from math import sqrt

def visualize_maze(matrix, bonus, start, end, route=None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    #1. Define walls and array of direction based on the route
    walls=[(i,j) for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j]=='x']

    if route:
        direction=[]
        for i in range(1,len(route)):
            if route[i][0]-route[i-1][0]>0:
                direction.append('v') #^
            elif route[i][0]-route[i-1][0]<0:
                direction.append('^') #v        
            elif route[i][1]-route[i-1][1]>0:
                direction.append('>')
            else:
                direction.append('<')

        direction.pop(0)

    #2. Drawing the map
    ax=plt.figure(dpi=100).add_subplot(111)

    for i in ['top','bottom','right','left']:
        ax.spines[i].set_visible(False)

    plt.scatter([i[1] for i in walls],[-i[0] for i in walls],
                marker='X',s=100,color='black')
    
    plt.scatter([i[1] for i in bonus],[-i[0] for i in bonus],
                marker='P',s=100,color='green')

    plt.scatter(start[1],-start[0],marker='*',
                s=100,color='gold')

    if route:
        for i in range(len(route)-2):
            plt.scatter(route[i+1][1],-route[i+1][0],
                        marker=direction[i],color='silver')

    plt.text(end[1],-end[0],'EXIT',color='red',
        horizontalalignment='center',
        verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')
    
    for _, point in enumerate(bonus):
        print(f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')

def read_file(file_name: str = 'maze.txt'):
    f=open(file_name,'r')
    n_bonus_points = int(next(f)[:-1])
    bonus_points = []
    for i in range(n_bonus_points):
        x, y, reward = map(int, next(f)[:-1].split(' '))
        bonus_points.append((x, y, reward))

    text=f.read()
    matrix=[list(i) for i in text.splitlines()]
    f.close()

    return bonus_points, matrix

bonus_points, matrix = read_file('bonus_map5.txt')

print(f'The height of the matrix: {len(matrix)}')
print(f'The width of the matrix: {len(matrix[0])}')

for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i][j]=='S':
            start=(i,j)

        elif matrix[i][j]==' ':
            if (i==0) or (i==len(matrix)-1) or (j==0) or (j==len(matrix[0])-1):
                end=(i,j)
                
        else:
            pass

# Graph lưu vị trí liền kề có thể đi tới được từ vị trí (i,j)
rows=len(matrix)
cols=len(matrix[0])
graph={}
for i in range(1,rows-1):
    for j in range(1,cols-1):
        if matrix[i][j]!='x':
            adj=[]
            for loc in [(i,j+1),(i,j-1),(i+1,j),(i-1,j)]:
                if matrix[loc[0]][loc[1]]!='x':
                    adj.append(loc)
            graph[(i,j)]=adj
if end[0]==0:
    graph[end] = [(end[0]+1,end[1])]
elif end[0]==rows-1:
    graph[end] = [(end[0]-1,end[1])]
elif end[1]==0:
    graph[end] = [(end[0],end[1]+1)]
else:
    graph[end] = [(end[0],end[1]-1)]

# Tìm kiếm mù
def dfs(graph, start, end):
    tracking={start:(0,0)}
    stack=[start]
    path=[]
    while len(stack):   # Còn phần tử trong stack
        current=stack.pop()     # Lấy phần tử cuối stack
        if current==end:    # Đã đến đích thì break
            break
        for neighbor in graph[current]:     # Xét 4 vị trí liền kề vị trí đang xét current
            if neighbor not in tracking:    # Nếu vị trí neighbor chưa được xét trước đó
                tracking[neighbor]=current  # Đánh dấu vị trí cha đi đến neighbor là current
                stack.append(neighbor)      # Thêm vị trí neighbor vào stack
    
    # Backtracking
    temp=end
    while temp!=(0,0):
        path.insert(0,tracking[temp])
        temp=tracking[temp]
    path.append(end)
    return path[1:]

def bfs(graph,start,end):
    queue=[]
    tracking={start:(0,0)}
    path=[]
    current=start
    while current!=end:     # Tìm khi vẫn chưa đến đích
        for neighbor in graph[current]:     # Xét 4 vị trí liền kề vị trí đang xét current
            if neighbor not in tracking:    # Nếu vị trí neighbor chưa được xét trước đó
                queue.append(neighbor)      # Thêm vị trí neighbor vào queue
                tracking[neighbor]=current  # Đánh dấu vị trí cha đi đến neighbor là current
        current=queue.pop(0)        # Lấy phần tử đầu queue
    
    # Backtracking
    temp=end
    while temp!=(0,0):
        path.insert(0,tracking[temp])
        temp=tracking[temp]
    path.append(end)
    return path[1:]

# Tìm kiếm có thông tin

# Hàm Heuristic - khoảng cách Manhattan: |end.X - pos.X| + |end.Y - pos.Y|
def heuristic(pos, end):
    return abs(pos[0]-end[0]) + abs(pos[1]-end[1])

# GBFS
def GBFS():
    heap = []       # Priority Queue lưu các biên, phần tử có dạng (h,(x,y)) -> h = heuristic của (x,y)
    parent = {start:(0,0)}    # Dict để truy vết: (x,y):(a,b) -> Nút cha đi tới (x,y) là (a,b)
    current = start
    while current!=end:         # Khi chưa đến đích
        for neighbor in graph[current]:     # Xét 4 vị trí liền kề vị trí đang xét current
            if neighbor not in parent:      # Nếu vị trí neighbor chưa được xét (chưa có nút nào đi tới)
                heappush(heap, (heuristic(neighbor, end),neighbor))      # Thêm neighbor vào heap
                parent[neighbor] = current      # Đánh dấu cha của neighbor là current
        current = heappop(heap)[1]     # Lấy phần tử đầu heap

    # Backtracking
    path = []
    track = end
    while track!=(0,0):
        path.insert(0, track)
        track = parent[track]
    return path

# A*    F(n) = G(n) + H(n)      -> G(n) = bước đi từ start đến n
def A_star():
    heap = []       # Priority Queue lưu các biên, phần tử có dạng (f,(x,y)) -> f = g+h
    # Dict lưu thông tin của (x,y):[g,h,(a,b)] -> thông tin g, h và nút cha (a,b) của (x,y)
    info = {start:[0,heuristic(start, end),(0,0)]}
    current = start

    while current!=end:         # Khi chưa đến đích
        for neighbor in graph[current]:     # Xét 4 vị trí liền kề vị trí đang xét current
            if neighbor not in info:      # Nếu vị trí neighbor chưa được xét (chưa có thông tin)
                info[neighbor] = [info[current][0]+1, heuristic(neighbor, end), current] # Thêm thông tin neighbor
                heappush(heap, (info[neighbor][0] + info[neighbor][1],neighbor))      # Thêm neighbor vào heap
            # Nếu neighbor đã được xét, cần kiểm tra lại g để tối ưu đường đi
            elif info[neighbor][0] > info[current][0] + 1:
                info[neighbor][0] = info[current][0] + 1    # Cập nhật lại g
                info[neighbor][2] = current                 # Cập nhật lại cha
        current = heappop(heap)[1]     # Lấy phần tử đầu heap

    # Backtracking
    path = []
    track = end
    while track!=(0,0):
        path.insert(0, track)
        track = info[track][2]
    return path

# Bonus Map
def find_path(begin, end, past_point):
    heap = [(0, begin)]
    info = {begin:[0,heuristic(begin, end),0,(0,0)]}
    way_out = False

    while (len(heap)):
        current = heappop(heap)[1]
        if current==end:
            way_out = True
            break
        for neighbor in graph[current]:
            if neighbor not in info:
                g = info[current][0] + 1
                h = heuristic(neighbor, end)
                b = 0
                if neighbor not in past_point:
                    for pts in bonus_points:
                        if pts[0]==neighbor[0] and pts[1]==neighbor[1]:
                            b = pts[2]
                            break
                info[neighbor] = [g, h, b, current]
                heappush(heap,(g+h+b, neighbor))
            elif info[neighbor][0] > info[current][0] + 1:
                info[neighbor][0] = info[current][0] + 1
                info[neighbor][3] = current

    path = []
    cost = 0
    track = end
    if way_out:
        while track!=(0,0):
            path.insert(0, track)
            cost += 1 + info[track][2]
            track = info[track][3]
        cost -= 1       # Dư 1 bước do đã ở sẵn vị trí begin chứ không phải bước đến từ (0,0)
    return [way_out, cost, path]

def solve_bonus_map():
    queue = [start]
    # short_matrix là dictionary lưu ma trận rút gọn, chỉ gồm các key là start, end và các bonus point
    # value là [chi phí từ start đến key [0], đường đi đến key [1] ]
    short_matrix = {(x[0],x[1]):[rows*cols, []] for x in bonus_points}
    short_matrix[start] = [0, [start]]
    short_matrix[end] = [rows*cols, []]

    while len(queue):
        current = queue.pop(0)
        for point in short_matrix:
            if point!= start and point!=current:
                wayout = find_path(current, point, short_matrix[current])
                if not wayout[0]:
                    continue
                if short_matrix[point][0] >= short_matrix[current][0] + wayout[1]:      # Nếu đường cũ lớn hơn đường đi mới thông qua current
                    short_matrix[point][0] = short_matrix[current][0] + wayout[1]       # Chi phí mới
                    short_matrix[point][1] = short_matrix[current][1] + wayout[2][1:]    # Đường đi mới bằng start -> current + current -> point
                    if point!=end:
                        queue.append(point)             # Thêm point vào queue để tiếp tục loang
    return short_matrix[end]

# wayoutDFS=dfs(graph, start, end)
# wayoutBFS=bfs(graph, start, end)
# visualize_maze(matrix,bonus_points,start,end,wayoutBFS)
# visualize_maze(matrix,bonus_points,start,end,wayoutDFS)
# sol_GBFS = GBFS()
# visualize_maze(matrix,bonus_points,start,end,sol_GBFS)
# sol_Astar = A_star()
# visualize_maze(matrix,bonus_points,start,end,sol_Astar)

#print(bonus_dict)
# wayout, cost, b_path = find_path(start, end, bonus_dict, True)
# print(wayout)
# print(cost)
# visualize_maze(matrix,bonus_points,start,end,b_path)

ans = solve_bonus_map()
print(ans[0])
visualize_maze(matrix,bonus_points,start,end,ans[1])