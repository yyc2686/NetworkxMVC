from service.ComplexNetworkService import ComplexNetworkService
from service.model.ComplexNetworkModel import ComplexNetworkModel

if __name__ == "__main__":
    # ----------------------------------------建立网络（按边建立）-------------------------------------------
    def createNetwork():
        networkService = ComplexNetworkService(
            ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)], isWeighted=True))


    # createNetwork()

    # ----------------------------------------测试BaseInfo（计算网络基本属性）-------------------------------------------
    def testBaseInfo():
        networkService = ComplexNetworkService(
            ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)], isWeighted=True))
        print(networkService.isDirected())
        print(networkService.neighbor())
        print(networkService.connectedComponents())
        print(networkService.adjacencyMatrix(isWeighted=False))
        print(networkService.adjacencyMatrix(isWeighted=True))

        diNetworkService = ComplexNetworkService(
            ComplexNetworkModel.generateDirectNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],
                                                      isWeighted=True))
        print(diNetworkService.isDirected())
        print(diNetworkService.inNeighbor())
        print(diNetworkService.outNeighbor())
        print(diNetworkService.convertNetworkFromDirectNetwork().connectedComponents())
        print(diNetworkService.adjacencyMatrix(isWeighted=False))
        print(diNetworkService.adjacencyMatrix(isWeighted=True))


    # testBaseInfo()

    # ----------------------------------------测试NetworkLibraryMetric（计算库函数指标）-------------------------------------------
    def testNetworkLibraryMetric():
        def networkExample():
            networkService = ComplexNetworkService(
                ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],
                                                    isWeighted=True))
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

        def diNetworkExample():
            diNetworkService = ComplexNetworkService(
                ComplexNetworkModel.generateDirectNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],
                                                          isWeighted=True))
            print(diNetworkService.inDegree())
            print(diNetworkService.outDegree())
            print(diNetworkService.inStrength())
            print(diNetworkService.outStrength())
            print(diNetworkService.pagerank(alpha=0.85, isWeighted=False))
            print(diNetworkService.pagerank(alpha=0.85, isWeighted=True))
            print(diNetworkService.clustering(isWeighted=False))
            print(diNetworkService.clustering(isWeighted=True))
            print(diNetworkService.averageClustering(isWeighted=False))
            print(diNetworkService.averageClustering(isWeighted=True))
            print(diNetworkService.inDegreeCentrality())
            print(diNetworkService.outDegreeCentrality())
            print(diNetworkService.closenessCentrality())
            print(diNetworkService.betweennessCentrality())
            print(diNetworkService.eigenvectorCentrality())
            # print(diNetworkService.degreeAssortativityCoefficient(isWeighted=False))
            # print(diNetworkService.degreeAssortativityCoefficient(isWeighted=True))
            print(diNetworkService.convertNetworkFromDirectNetwork().diameter())
            print(diNetworkService.averagePathLength())
            print(diNetworkService.convertNetworkFromDirectNetwork().averagePathLength())

        networkExample()
        diNetworkExample()


    # testNetworkLibraryMetric()

    # ----------------------------------------测试NetworkCommonMetric（计算自定义函数指标）-------------------------------------------
    def testNetworkCommonMetric():
        def networkExample():
            networkService = ComplexNetworkService(
                ComplexNetworkModel.generateNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],
                                                    isWeighted=True))
            print(networkService.hIndex())
            print(networkService.h2())
            print(networkService.CI())
            print(networkService.core(isWeighted=False))
            print(networkService.core(isWeighted=True))

        def diNetworkExample():
            diNetworkService = ComplexNetworkService(
                ComplexNetworkModel.generateDirectNetwork(edges=[(1, 2, 1), (2, 3, 2), (3, 1, 3), (3, 4, 4)],
                                                          isWeighted=True))
            print(diNetworkService.hIndex(type="in"))
            print(diNetworkService.hIndex(type="out"))
            print(diNetworkService.h2(type="in"))
            print(diNetworkService.h2(type="out"))
            print(diNetworkService.CIIn())
            print(diNetworkService.CIOut())
            print(diNetworkService.inCore(isWeighted=False))
            print(diNetworkService.inCore(isWeighted=True))
            print(diNetworkService.outCore(isWeighted=False))
            print(diNetworkService.outCore(isWeighted=True))

        networkExample()
        diNetworkExample()


    # testNetworkCommonMetric()
    pass
