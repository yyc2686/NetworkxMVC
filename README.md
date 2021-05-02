## NetworkxMVC快速使用说明

---

### 前言
科研工作中，涉及复杂网络的相关学科，往往离不开Python的Networkx库。作为一个优秀的第三方库，networkx集成了大量常用的方法，包含网络构建，网络拓扑属性计算，网络指标计算，以及经典网络算法实现等，完整教程可参考：
- [中文教程](https://www.osgeo.cn/networkx/tutorial.html)
- [英文教程](https://networkx.org/documentation/stable/tutorial.html)

但科研工作中，我们常用的功能其实不多，“乱花渐欲迷人眼”，或许整合常用的一些方法对于绝大多数人而言可能更适用。

本项目基于Networkx作简单的包装，旨在借鉴MVC的部分设计思路整合常用的Networkx方法，封装网络构建、网络拓扑属性、自带网络指标和自定义指标四个模块。项目的目标一方面是帮助快速上手Networkx，用于网络分析与指标计算；另一方面，通过MVC的功能拆分，以及自定义指标模块的设定，降低代码耦合度的同时也希望抛砖引玉，帮助拓展更多常用的方法。

使用前，根据requirements.txt，安装所需全部库函数。

使用时，导入service下的两个类ComplexNetworkService、ComplexNetworkModel

```python
from service.ComplexNetworkService import ComplexNetworkService
from service.model.ComplexNetworkModel import ComplexNetworkModel
```

计算网络属性和指标前，需创建好一个网络对象，然后直接使用网络对象调用对应方法。

项目链接：[https://github.com/yyc2686/NetworkxMVC](https://github.com/yyc2686/NetworkxMVC)



### 一、建立网络

```python
from service.ComplexNetworkService import ComplexNetworkService
from service.model.ComplexNetworkModel import ComplexNetworkModel

# 创建无向含权网络
networkService = ComplexNetworkService(
            ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)], isWeighted=True))

# 创建有向含权网络
diNetworkService = ComplexNetworkService(
            ComplexNetworkModel.generateDirectNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],isWeighted=True))
```

目前仅支持按边集创建网络，generateNetwork()和generateDirectNetwork()分别创建无向和有向网络，并通过isWeighted指定网络是否含权。



### 二、获取网络拓扑属性

```python
# 构建无向含权网络
networkService = ComplexNetworkService(
    ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)], isWeighted=True))

# 计算网络拓扑属性
print(networkService.isDirected())
print(networkService.neighbor())
print(networkService.connectedComponents())
print(networkService.adjacencyMatrix(isWeighted=False))
print(networkService.adjacencyMatrix(isWeighted=True))
```



### 三、计算networkx自带的网络指标

```python
from service.ComplexNetworkService import ComplexNetworkService
from service.model.ComplexNetworkModel import ComplexNetworkModel

# 创建无向网络
networkService = ComplexNetworkService(
    ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)], isWeighted=True))

# 计算无向网络库指标
print(networkService.degree())
print(networkService.strength())
print(networkService.pagerank(alpha=0.85, isWeighted=False))
print(networkService.pagerank(alpha=0.85, isWeighted=True))
print(networkService.clustering(isWeighted=False))
print(networkService.clustering(isWeighted=True))
print(networkService.averageClustering(isWeighted=False))
print(networkService.averageClustering(isWeighted=True))
print(networkService.degreeCentrality())
print(networkService.closenessCentrality())
print(networkService.betweennessCentrality())
print(networkService.eigenvectorCentrality())
print(networkService.degreeAssortativityCoefficient(isWeighted=False))
print(networkService.degreeAssortativityCoefficient(isWeighted=True))
print(networkService.diameter())
print(networkService.averagePathLength())
```



### 四、计算自定义网络指标

```python
from service.ComplexNetworkService import ComplexNetworkService
from service.model.ComplexNetworkModel import ComplexNetworkModel

# 创建无向网络
networkService = ComplexNetworkService(
    ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],isWeighted=True))

# 计算无向网络自定义指标
print(networkService.hIndex())
print(networkService.h2())
print(networkService.CI())
print(networkService.core(isWeighted=False))
print(networkService.core(isWeighted=True))
```

若想要拓展其他自定义指标，仅需在service.ComplexNetworkService中的NetworkCustomMetric类中，编写自定义的方法即可，如：

```python
@classmethod
def function(cls, isWeighted=False):
    """建议写上详细的注释"""

    try:
		"""自定义方法主体"""
    except Exception as errMsg:
        raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)
```


