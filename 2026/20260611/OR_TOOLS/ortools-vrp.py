## 包的导入
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import tsplib95
import matplotlib.pyplot as plt
## 设定数据目录并读入数据
fname: str = "gr120.tsp"
input_fpath: str = r"data/" + fname
data = tsplib95.load(input_fpath).as_name_dict()

## 建立模型
# 创建routing index manager
n_vehicle = 1 # 车辆数，在TSP问题中为1
depot_index = 0 # 车库，设置为0号节点
manager = pywrapcp.RoutingIndexManager(data['dimension'], n_vehicle, depot_index)
# 创建一个Routing model
routing = pywrapcp.RoutingModel(manager)
# 创建一个distance call back，这里使用欧几里得距离
def distance_callback(from_index: int, to_index: int):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    # +1是因为读入数据时节点下标从1开始
    from_coor = data['display_data'][from_node+1]
    to_coor = data['display_data'][to_node+1]
    return (to_coor[0] - from_coor[0])**2 + (to_coor[1] - from_coor[1])**2
 # 因为我们需要在车辆经过节点i是为其在距离维度d上累加一定的值，所以这里我们使用transitCallBack; 之后我们会看到在设置时间窗口时，会用到cumul类的维度变量
transit_callback_index = routing.RegisterTransitCallback(distance_callback)
# 给每条边设置cost
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
# 设置求解的启发式算法
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
# 求解
solution = routing.SolveWithParameters(search_parameters)

## 输出结果：
if solution:
    print("路径总长度: {} miles".format(solution.ObjectiveValue()))
    index = routing.Start(0) # routing.Start(0)代表获取0号车辆的起点
    path: str = "旅行商的路径：\n"
    visiting_sequence: list = []
    while not routing.IsEnd(index):
        visiting_sequence.append(manager.IndexToNode(index)+1)
        path += " {} ->".format(manager.IndexToNode(index)+1)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
    visiting_sequence.append(manager.IndexToNode(index)+1)
    path += " {}\n".format(manager.IndexToNode(index)+1)
    print(path)
else:
    print("没有得到可行解")

plt.figure(figsize=(8, 6))
# 画出depot
depot_coor = data['display_data'][depot_index + 1]
plt.plot(depot_coor[0], depot_coor[1], 'r*', markersize=11)
# 路径可视化
for i, j in zip(visiting_sequence, visiting_sequence[1:]):
    start_coor = data['display_data'][i]
    end_coor = data['display_data'][j]
    plt.arrow(start_coor[0], start_coor[1], end_coor[0] - start_coor[0], end_coor[1] - start_coor[1])
plt.xlabel("X coordinate", fontsize = 14)
plt.ylabel("Y coordinate", fontsize = 14)
plt.title("TSP path for {}".format(fname), fontsize = 16)