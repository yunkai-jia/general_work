## Ortools的VRP求解器简介
### 参考网址
https://zhuanlan.zhihu.com/p/405467844

### 谷歌的Ortools整合了许多对运筹优化问题的求解器，其中最好用的部分就是VRP求解器。

在ortools中，VRP求解器是建立在constraint programming求解器之上的，因此除了一些经典的VRP问题约束，例如最大负载，时间窗以外，还可以通过约束规划求解器，灵活地添加一系列高度定制化的约束，例如要求某辆车经过某个节点，要求一个节点被多台车访问等。

在ortools中求解VRP问题时，最常用的有两类变量，我们可以从抽象层面了解一下其含义：

路径变量(Path variable)：
next(i) - 代表节点 i 在路径中的后继节点的下标。用IndexToNode()可以从下表获取节点变量。
vehicle(i) - 表示节点 i 所属于的车辆路径编号，即第 i 个节点由哪辆车来服务。
active(i) - 布尔变量，如果节点 i 被访问，那么值为True，否则为False。
下面的几组关系是等价的：active(i) == 0 <==> next(i) == i <==> vehicle(i) == -1，next(i) == j ==> vehicle(j) == vehicle(i)
维度变量(Dimension variable) - 用于标记经过路径时累加的值，例如车辆的负载、消耗时间、行驶距离等。常用的有两类：
cumul(i, d) - 代表车辆到达节点 i 时维度 d 的值
transit(i, d) - 代表车辆经过节点 i 时，需要在维度 d 上累加的值
这两者的关系为： next(i)==j ==> cumul(j,d)=cumul(i,d)+transit(i,d)
因此，从抽象层面上来讲，每台车从depot出发，不断进行next(i)操作，最终回到depot，即可求得该车辆的路径。每次next(i)操作的同时，我们也通过transit(i,d)不断累加我们关注的值，例如车辆的总行驶距离、消耗时间等。这就是ortools对VRP问题的抽象概括。