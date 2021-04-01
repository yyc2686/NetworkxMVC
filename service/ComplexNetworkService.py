import networkx as nx
from utils.CommonUtils import CommonUtils
from error.BusinessException import BusinessException
from error.EmBusinessError import EmBusinessError


class Network:
    network = None

    @classmethod
    def setNetwork(cls, complexNetworkModel):
        cls.network = complexNetworkModel.network


class NetworkBaseInfo:
    """查询网络基本属性"""

    @classmethod
    def networkNodes(cls):
        """网络所有节点"""
        return cls.network.nodes

    @classmethod
    def networkEdges(cls):
        """网络所有边"""
        return cls.network.edges

    @classmethod
    def networkDensity(cls):
        """网络密度"""
        N, L = len(cls.networkNodes()), len(cls.networkEdges())
        return 2 * L / (N * (N - 1))

    @classmethod
    def isDirected(cls):
        """Returns True if graph is directed, False otherwise."""
        return cls.network.is_directed()

    @classmethod
    def neighbor(cls):
        """
        计算无向网络节点的邻居
        :return: dict
        """
        if cls.isDirected():
            raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "有向网络应使用inNeighbor()或者outNeighbor()")
        else:
            return {node: dict(cls.network[node]) for node in cls.network}

    @classmethod
    def inNeighbor(cls):
        """
        计算有向网络节点的入邻居
        :return: dict
        """
        if cls.isDirected():
            return {node: {n: cls.network[n][node] for n in cls.network if node in cls.network[n]} for node in
                    cls.network}
        else:
            raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用neighbor()")

    @classmethod
    def outNeighbor(cls):
        if cls.isDirected():
            return {node: dict(cls.network[node]) for node in cls.network}
        else:
            raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用neighbor()")

    @classmethod
    def connectedComponents(cls, **kwargs):
        """Generate connected components.

        Parameters
        ----------
        G : NetworkX graph
           An undirected graph

        Returns
        -------
        comp : generator of sets
           A generator of sets of nodes, one for each component of G.

        Raises
        ------
        NetworkXNotImplemented:
            If G is directed.

        Examples
        --------
        Generate a sorted list of connected components, largest first.

        >>> G = nx.path_graph(4)
        >>> nx.add_path(G, [10, 11, 12])
        >>> [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        [4, 3]

        If you only want the largest connected component, it's more
        efficient to use max instead of sort.

        >>> largest_cc = max(nx.connected_components(G), key=len)

        To create the induced subgraph of each component use:
        >>> S = [G.subgraph(c).copy() for c in connected_components(G)]

        See Also
        --------
        strongly_connected_components
        weakly_connected_components

        Notes
        -----
        For undirected graphs only.

        """

        if cls.isDirected():
            raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR,
                                    "有向网络应先使用convertNetworkFromDirectNetwork()转换成无向网络，再计算最大连通集团")

        if kwargs.get('max'):
            return max(nx.connected_components(cls.network), key=len)  # 高效找出最大的联通成分，其实就是sorted里面的No.1
        else:
            return nx.connected_components(cls.network)

    @classmethod
    def adjacencyMatrix(cls, isWeighted=False):
        import numpy as np
        return np.array(nx.adjacency_matrix(cls.network, weight="weight" if isWeighted else "").todense())

    @classmethod
    def convertNetworkFromDirectNetwork(cls):
        cls.network = cls.network.to_undirected()
        return cls


class NetworkLibraryMetric:
    """查询网络库指标"""

    # 度 ---------------------------------------------------------------------------------------------------------
    @classmethod
    def degree(cls):
        """
        :return: 无向网络度，dict
        """
        try:
            if cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "有向网络应使用inDegree()或者outDegree()")
            else:
                return CommonUtils.orderDict({record[0]: record[1] for record in cls.network.degree}, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def inDegree(cls):
        """
        :return: 有向网络度入度，dict
        """

        try:
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用degree()")
            else:
                return CommonUtils.orderDict({record[0]: record[1] for record in cls.network.in_degree}, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def outDegree(cls):
        """
        :return: 有向网络度出度，dict
        """

        try:
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用degree()")
            else:
                return CommonUtils.orderDict({record[0]: record[1] for record in cls.network.out_degree}, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # 强度(含权度) -----------------------------------------------------------------------------------------------
    @classmethod
    def strength(cls):
        """
        :return: 无向网络强度，dict
        """

        try:
            if cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "有向网络应使用inStrength()或者outStrength()")
            else:
                return CommonUtils.orderDict({record[0]: record[1] for record in cls.network.degree(weight='weight')},
                                             index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def inStrength(cls):
        """
        :return: 有向网络度入强度，dict
        """

        try:
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用strength()")
            else:
                return CommonUtils.orderDict(
                    {record[0]: record[1] for record in cls.network.in_degree(weight='weight')}, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def outStrength(cls):
        """
        :return: 有向网络度出强度，dict
        """
        try:
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用strength()")
            else:
                return CommonUtils.orderDict(
                    {record[0]: record[1] for record in cls.network.out_degree(weight='weight')}, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # Link Analysis ----------------------------------------------------------------------------------------------
    @classmethod
    def pagerank(cls, alpha=0.85, isWeighted=False):
        """Returns the PageRank of the nodes in the graph.

            PageRank computes a ranking of the nodes in the graph_ G based on
            the structure of the incoming links. It was originally designed as
            an algorithm to rank web pages.

            Parameters
            ----------
            G : graph
              A NetworkX graph.  Undirected graphs will be converted to a directed
              graph with two directed edges for each undirected edge.

            alpha : float, optional
              Damping parameter for PageRank, @classmethod
    default=0.85.

            personalization: dict, optional
              The "personalization vector" consisting of a dictionary with a
              key some subset of graph nodes and personalization value each of those.
              At least one personalization value must be non-zero.
              If not specfiied, a nodes personalization value will be zero.
              By default, a uniform distribution is used.

            max_iter : integer, optional
              Maximum number of iterations in power method eigenvalue solver.

            tol : float, optional
              Error tolerance used to check convergence in power method solver.

            nstart : dictionary, optional
              Starting value of PageRank iteration for each node.

            weight : key, optional
              Edge data key to use as weight.  If None weights are set to 1.

            dangling: dict, optional
              The outedges to be assigned to any "dangling" nodes, i.e., nodes without
              any outedges. The dict key is the node the outedge points to and the dict
              value is the weight of that outedge. By default, dangling nodes are given
              outedges according to the personalization vector (uniform if not
              specified). This must be selected to result in an irreducible transition
              matrix (see notes under google_matrix). It may be common to have the
              dangling dict to be the same as the personalization dict.

            Returns
            -------
            pagerank : dictionary
               Dictionary of nodes with PageRank as value

            Examples
            --------
            >>> G = nx.DiGraph(nx.path_graph(4))
            >>> pr = nx.pagerank(G, alpha=0.9)

            Notes
            -----
            The eigenvector calculation is done by the power iteration method
            and has no guarantee of convergence.  The iteration will stop after
            an error tolerance of ``len(G) * tol`` has been reached. If the
            number of iterations exceed `max_iter`, a
            :exc:`networkx.exception.PowerIterationFailedConvergence` exception
            is raised.

            The PageRank algorithm was designed for directed graphs but this
            algorithm does not check if the input graph is directed and will
            execute on undirected graphs by converting each edge in the
            directed graph to two edges.

            See Also
            --------
            pagerank_numpy, pagerank_scipy, google_matrix

            Raises
            ------
            PowerIterationFailedConvergence
                If the algorithm fails to converge to the specified tolerance
                within the specified number of iterations of the power iteration
                method.

            References
            ----------
            .. [1] A. Langville and C. Meyer,
               "A survey of eigenvector methods of web information retrieval."
               http://citeseer.ist.psu.edu/713792.html
            .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
               The PageRank citation ranking: Bringing order to the Web. 1999
               http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf

            """
        try:
            if isWeighted:
                return CommonUtils.orderDict(nx.pagerank(cls.network, alpha=alpha), index=1)
            else:
                return CommonUtils.orderDict(nx.pagerank(cls.network, alpha=alpha, weight=False), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # Cluster ----------------------------------------------------------------------------------------------------
    @classmethod
    def clustering(cls, isWeighted=False, **kwargs):
        r"""Compute the clustering coefficient for nodes.

    For unweighted graphs, the clustering of a node :math:`u`
    is the fraction of possible triangles through that node that exist,

    .. math::

      c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},

    where :math:`T(u)` is the number of triangles through node :math:`u` and
    :math:`deg(u)` is the degree of :math:`u`.

    For weighted graphs, there are several ways to define clustering [1]_.
    the one used here is defined
    as the geometric average of the subgraph edge weights [2]_,

    .. math::

       c_u = \frac{1}{deg(u)(deg(u)-1))}
             \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

    The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
    in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

    The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

    For directed graphs, the clustering is similarly defined as the fraction
    of all possible directed triangles or geometric average of the subgraph
    edge weights for unweighted and weighted directed graph respectively [3]_.

    .. math::

       c_u = \frac{1}{deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u)}
             T(u),

    where :math:`T(u)` is the number of directed triangles through node
    :math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
    :math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
    :math:`u`.

    Parameters
    ----------
    G : graph

    nodes : container of nodes, optional (default=all nodes in G)
       Compute clustering for nodes in this container.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    out : float, or dictionary
       Clustering coefficient at specified nodes

    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(nx.clustering(G,0))
    1.0
    >>> print(nx.clustering(G))
    {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    Notes
    -----
    cls loops are ignored.

    References
    ----------
    .. [1] Generalizations of the clustering coefficient to weighted
       complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
       K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
       http://jponnela.com/web_documents/a9.pdf
    .. [2] Intensity and coherence of motifs in weighted complex
       networks by J. P. Onnela, J. Saramäki, J. Kertész, and K. Kaski,
       Physical Review E, 71(6), 065103 (2005).
    .. [3] Clustering in complex directed networks by G. Fagiolo,
       Physical Review E, 76(2), 026107 (2007).
    """

        try:
            return CommonUtils.orderDict(
                nx.clustering(cls.network, nodes=kwargs.get('nodes'), weight="weight" if isWeighted else ""), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def averageClustering(cls, isWeighted=False, **kwargs):
        r"""Compute the average clustering coefficient for the graph G.

            The clustering coefficient for the graph is the average,

            .. math::

               C = \frac{1}{n}\sum_{v \in G} c_v,

            where :math:`n` is the number of nodes in `G`.

            Parameters
            ----------
            G : graph

            nodes : container of nodes, optional (default=all nodes in G)
               Compute average clustering for nodes in this container.

            weight : string or None, optional (default=None)
               The edge attribute that holds the numerical value used as a weight.
               If None, then each edge has weight 1.

            count_zeros : bool
               If False include only the nodes with nonzero clustering in the average.

            Returns
            -------
            avg : float
               Average clustering

            Examples
            --------
            >>> G=nx.complete_graph(5)
            >>> print(nx.average_clustering(G))
            1.0

            Notes
            -----
            This is a space saving routine; it might be faster
            to use the clustering function to get a list and then take the average.

            cls loops are ignored.

            References
            ----------
            .. [1] Generalizations of the clustering coefficient to weighted
               complex networks by J. Saramäki, M. Kivelä, J.-P. Onnela,
               K. Kaski, and J. Kertész, Physical Review E, 75 027105 (2007).
               http://jponnela.com/web_documents/a9.pdf
            .. [2] Marcus Kaiser,  Mean clustering coefficients: the role of isolated
               nodes and leafs on clustering measures for small-world networks.
               https://arxiv.org/abs/0802.2512
            """

        try:
            return nx.average_clustering(cls.network, nodes=kwargs.get('nodes'), weight="weight" if isWeighted else "",
                                         count_zeros=kwargs.get('count_zeros', True))
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # Centrality -------------------------------------------------------------------------------------------------
    @classmethod
    def degreeCentrality(cls):
        """Compute the degree centrality for nodes.

        The degree centrality for a node v is the fraction of nodes it
        is connected to.

        Parameters
        ----------
        G : graph
          A networkx graph

        Returns
        -------
        nodes : dictionary
           Dictionary of nodes with degree centrality as the value.

        See Also
        --------
        betweenness_centrality, load_centrality, eigenvector_centrality

        Notes
        -----
        The degree centrality values are normalized by dividing by the maximum
        possible degree in a simple graph n-1 where n is the number of nodes in G.

        For multigraphs or graphs with cls loops the maximum degree might
        be higher than n-1 and values of degree centrality greater than 1
        are possible.
        """

        try:
            if cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR,
                                        "有向网络应使用inDegreeCentrality()或者outDegreeCentrality()")
            else:
                return CommonUtils.orderDict(nx.degree_centrality(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def inDegreeCentrality(cls):
        """Compute the in-degree centrality for nodes.

        The in-degree centrality for a node v is the fraction of nodes its
        incoming edges are connected to.

        Parameters
        ----------
        G : graph
            A NetworkX graph

        Returns
        -------
        nodes : dictionary
            Dictionary of nodes with in-degree centrality as values.

        Raises
        ------
        NetworkXNotImplemented:
            If G is undirected.

        See Also
        --------
        degree_centrality, out_degree_centrality

        Notes
        -----
        The degree centrality values are normalized by dividing by the maximum
        possible degree in a simple graph n-1 where n is the number of nodes in G.

        For multigraphs or graphs with cls loops the maximum degree might
        be higher than n-1 and values of degree centrality greater than 1
        are possible.
        """
        try:
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用degreeCentrality()")
            else:
                return CommonUtils.orderDict(nx.in_degree_centrality(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def outDegreeCentrality(cls):
        """Compute the out-degree centrality for nodes.

            The out-degree centrality for a node v is the fraction of nodes its
            outgoing edges are connected to.

            Parameters
            ----------
            G : graph
                A NetworkX graph

            Returns
            -------
            nodes : dictionary
                Dictionary of nodes with out-degree centrality as values.

            Raises
            ------
            NetworkXNotImplemented:
                If G is undirected.

            See Also
            --------
            degree_centrality, in_degree_centrality

            Notes
            -----
            The degree centrality values are normalized by dividing by the maximum
            possible degree in a simple graph n-1 where n is the number of nodes in G.

            For multigraphs or graphs with cls loops the maximum degree might
            be higher than n-1 and values of degree centrality greater than 1
            are possible.
            """

        try:
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应使用degreeCentrality()")
            else:
                return CommonUtils.orderDict(nx.out_degree_centrality(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def closenessCentrality(cls):
        r"""Compute closeness centrality for nodes.

            Closeness centrality [1]_ of a node `u` is the reciprocal of the
            average shortest path distance to `u` over all `n-1` reachable nodes.

            .. math::

                C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

            where `d(v, u)` is the shortest-path distance between `v` and `u`,
            and `n` is the number of nodes that can reach `u`. Notice that the
            closeness distance function computes the incoming distance to `u`
            for directed graphs. To use outward distance, act on `G.reverse()`.

            Notice that higher values of closeness indicate higher centrality.

            Wasserman and Faust propose an improved formula for graphs with
            more than one connected component. The result is "a ratio of the
            fraction of actors in the group who are reachable, to the average
            distance" from the reachable actors [2]_. You might think this
            scale factor is inverted but it is not. As is, nodes from small
            components receive a smaller closeness value. Letting `N` denote
            the number of nodes in the graph,

            .. math::

                C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

            Parameters
            ----------
            G : graph
              A NetworkX graph

            u : node, optional
              Return only the value for node u

            distance : edge attribute key, optional (default=None)
              Use the specified edge attribute as the edge distance in shortest
              path calculations

            wf_improved : bool, optional (default=True)
              If True, scale by the fraction of nodes reachable. This gives the
              Wasserman and Faust improved formula. For single component graphs
              it is the same as the original formula.

            Returns
            -------
            nodes : dictionary
              Dictionary of nodes with closeness centrality as the value.

            See Also
            --------
            betweenness_centrality, load_centrality, eigenvector_centrality,
            degree_centrality, incremental_closeness_centrality

            Notes
            -----
            The closeness centrality is normalized to `(n-1)/(|G|-1)` where
            `n` is the number of nodes in the connected part of graph
            containing the node.  If the graph is not completely connected,
            this algorithm computes the closeness centrality for each
            connected part separately scaled by that parts size.

            If the 'distance' keyword is set to an edge attribute key then the
            shortest-path length will be computed using Dijkstra's algorithm with
            that edge attribute as the edge weight.

            The closeness centrality uses *inward* distance to a node, not outward.
            If you want to use outword distances apply the function to `G.reverse()`

            In NetworkX 2.2 and earlier a bug caused Dijkstra's algorithm to use the
            outward distance rather than the inward distance. If you use a 'distance'
            keyword and a DiGraph, your results will change between v2.2 and v2.3.

            References
            ----------
            .. [1] Linton C. Freeman: Centrality in networks: I.
               Conceptual clarification. Social Networks 1:215-239, 1979.
               http://leonidzhukov.ru/hse/2013/socialnetworks/papers/freeman79-centrality.pdf
            .. [2] pg. 201 of Wasserman, S. and Faust, K.,
               Social Network Analysis: Methods and Applications, 1994,
               Cambridge University Press.
            """
        try:
            return CommonUtils.orderDict(nx.closeness_centrality(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def betweennessCentrality(cls):
        r"""Compute the shortest-path betweenness centrality for nodes.

            Betweenness centrality of a node $v$ is the sum of the
            fraction of all-pairs shortest paths that pass through $v$

            .. math::

               c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

            where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
            shortest $(s, t)$-paths,  and $\sigma(s, t|v)$ is the number of
            those paths  passing through some  node $v$ other than $s, t$.
            If $s = t$, $\sigma(s, t) = 1$, and if $v \in {s, t}$,
            $\sigma(s, t|v) = 0$ [2]_.

            Parameters
            ----------
            G : graph
              A NetworkX graph.

            k : int, optional (default=None)
              If k is not None use k node samples to estimate betweenness.
              The value of k <= n where n is the number of nodes in the graph.
              Higher values give better approximation.

            normalized : bool, optional
              If True the betweenness values are normalized by `2/((n-1)(n-2))`
              for graphs, and `1/((n-1)(n-2))` for directed graphs where `n`
              is the number of nodes in G.

            weight : None or string, optional (default=None)
              If None, all edge weights are considered equal.
              Otherwise holds the name of the edge attribute used as weight.

            endpoints : bool, optional
              If True include the endpoints in the shortest path counts.

            seed : integer, random_state, or None (default)
                Indicator of random number generation state.
                See :ref:`Randomness<randomness>`.
                Note that this is only used if k is not None.

            Returns
            -------
            nodes : dictionary
               Dictionary of nodes with betweenness centrality as the value.

            See Also
            --------
            edge_betweenness_centrality
            load_centrality

            Notes
            -----
            The algorithm is from Ulrik Brandes [1]_.
            See [4]_ for the original first published version and [2]_ for details on
            algorithms for variations and related metrics.

            For approximate betweenness calculations set k=#samples to use
            k nodes ("pivots") to estimate the betweenness values. For an estimate
            of the number of pivots needed see [3]_.

            For weighted graphs the edge weights must be greater than zero.
            Zero edge weights can produce an infinite number of equal length
            paths between pairs of nodes.

            The total number of paths between source and target is counted
            differently for directed and undirected graphs. Directed paths
            are easy to count. Undirected paths are tricky: should a path
            from "u" to "v" count as 1 undirected path or as 2 directed paths?

            For betweenness_centrality we report the number of undirected
            paths when G is undirected.

            For betweenness_centrality_subset the reporting is different.
            If the source and target subsets are the same, then we want
            to count undirected paths. But if the source and target subsets
            differ -- for example, if sources is {0} and targets is {1},
            then we are only counting the paths in one direction. They are
            undirected paths but we are counting them in a directed way.
            To count them as undirected paths, each should count as half a path.

            References
            ----------
            .. [1] Ulrik Brandes:
               A Faster Algorithm for Betweenness Centrality.
               Journal of Mathematical Sociology 25(2):163-177, 2001.
               http://www.inf.uni-konstanz.de/algo/publications/b-fabc-01.pdf
            .. [2] Ulrik Brandes:
               On Variants of Shortest-Path Betweenness
               Centrality and their Generic Computation.
               Social Networks 30(2):136-145, 2008.
               http://www.inf.uni-konstanz.de/algo/publications/b-vspbc-08.pdf
            .. [3] Ulrik Brandes and Christian Pich:
               Centrality Estimation in Large Networks.
               International Journal of Bifurcation and Chaos 17(7):2303-2318, 2007.
               http://www.inf.uni-konstanz.de/algo/publications/bp-celn-06.pdf
            .. [4] Linton C. Freeman:
               A set of measures of centrality based on betweenness.
               Sociometry 40: 35–41, 1977
               http://moreno.ss.uci.edu/23.pdf
            """

        try:
            return CommonUtils.orderDict(nx.betweenness_centrality(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    @classmethod
    def eigenvectorCentrality(cls):
        r"""Compute the eigenvector centrality for the graph `G`.

           Eigenvector centrality computes the centrality for a node based on the
           centrality of its neighbors. The eigenvector centrality for node $i$ is
           the $i$-th element of the vector $x$ defined by the equation

           .. math::

               Ax = \lambda x

           where $A$ is the adjacency matrix of the graph `G` with eigenvalue
           $\lambda$. By virtue of the Perron–Frobenius theorem, there is a unique
           solution $x$, all of whose entries are positive, if $\lambda$ is the
           largest eigenvalue of the adjacency matrix $A$ ([2]_).

           Parameters
           ----------
           G : graph
             A networkx graph

           max_iter : integer, optional (default=100)
             Maximum number of iterations in power method.

           tol : float, optional (default=1.0e-6)
             Error tolerance used to check convergence in power method iteration.

           nstart : dictionary, optional (default=None)
             Starting value of eigenvector iteration for each node.

           weight : None or string, optional (default=None)
             If None, all edge weights are considered equal.
             Otherwise holds the name of the edge attribute used as weight.

           Returns
           -------
           nodes : dictionary
              Dictionary of nodes with eigenvector centrality as the value.

           Examples
           --------
           >>> G = nx.path_graph(4)
           >>> centrality = nx.eigenvector_centrality(G)
           >>> sorted((v, '{:0.2f}'.format(c)) for v, c in centrality.items())
           [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

           Raises
           ------
           NetworkXPointlessConcept
               If the graph `G` is the null graph.

           NetworkXError
               If each value in `nstart` is zero.

           PowerIterationFailedConvergence
               If the algorithm fails to converge to the specified tolerance
               within the specified number of iterations of the power iteration
               method.

           See Also
           --------
           eigenvector_centrality_numpy
           pagerank
           hits

           Notes
           -----
           The measure was introduced by [1]_ and is discussed in [2]_.

           The power iteration method is used to compute the eigenvector and
           convergence is **not** guaranteed. Our method stops after ``max_iter``
           iterations or when the change in the computed vector between two
           iterations is smaller than an error tolerance of
           ``G.number_of_nodes() * tol``. This implementation uses ($A + I$)
           rather than the adjacency matrix $A$ because it shifts the spectrum
           to enable discerning the correct eigenvector even for networks with
           multiple dominant eigenvalues.

           For directed graphs this is "left" eigenvector centrality which corresponds
           to the in-edges in the graph. For out-edges eigenvector centrality
           first reverse the graph with ``G.reverse()``.

           References
           ----------
           .. [1] Phillip Bonacich.
              "Power and Centrality: A Family of Measures."
              *American Journal of Sociology* 92(5):1170–1182, 1986
              <http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf>
           .. [2] Mark E. J. Newman.
              *Networks: An Introduction.*
              Oxford University Press, USA, 2010, pp. 169.

           """

        try:
            return CommonUtils.orderDict(nx.eigenvector_centrality(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # 同配系数 ---------------------------------------------------------------------------------------------------
    @classmethod
    def degreeAssortativityCoefficient(cls, x='in', y='in', isWeighted=False, pearson=True, **kwargs):
        """Compute degree assortativity of graph.

    Assortativity measures the similarity of connections
    in the graph with respect to the node degree.

    Parameters
    ----------
    G : NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Compute degree assortativity only for nodes in container.
        The default is all nodes.

    Returns
    -------
    r : float
       Assortativity of graph by degree.

    Examples
    --------
    >>> G=nx.path_graph(4)
    >>> r=nx.degree_assortativity_coefficient(G)
    >>> print("%3.1f"%r)
    -0.5

    See Also
    --------
    attribute_assortativity_coefficient
    numeric_assortativity_coefficient
    neighbor_connectivity
    degree_mixing_dict
    degree_mixing_matrix

    Notes
    -----
    This computes Eq. (21) in Ref. [1]_ , where e is the joint
    probability distribution (mixing matrix) of the degrees.  If G is
    directed than the matrix e is the joint probability of the
    user-specified degree type for the source and target.

    References
    ----------
    .. [1] M. E. J. Newman, Mixing patterns in networks,
       Physical Review E, 67 026126, 2003
    .. [2] Foster, J.G., Foster, D.V., Grassberger, P. & Paczuski, M.
       Edge direction and the structure of networks, PNAS 107, 10815-20 (2010).
    """
        try:
            if pearson:
                return nx.degree_pearson_correlation_coefficient(cls.network, x=x, y=y,
                                                                 weight="weight" if isWeighted else "",
                                                                 nodes=kwargs.get('nodes'))
            else:
                return nx.degree_assortativity_coefficient(cls.network, x=x, y=y, weight="weight" if isWeighted else "",
                                                           nodes=kwargs.get('nodes'))
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # 网络直径 -------------------------------------------------------------------------------------------------------
    @classmethod
    def diameter(cls):
        try:
            return nx.diameter(cls.network)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)

    # 网络平均最短路径长度 -------------------------------------------------------------------------------------------
    @classmethod
    def averagePathLength(cls):
        try:
            return nx.average_shortest_path_length(cls.network)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_LIBRARY_METRIC_ERROR, errMsg)


class NetworkCustomMetric:
    """查询网络自定义指标"""

    # h-index ----------------------------------------------------------------------------------------------------
    @classmethod
    def hIndex(cls, type=""):
        """
        h-index ，又称为h指数或h因子（h-factor），是一种评价学术成就的新方法。
        h代表“高引用次数”（high citations），一名科研人员的h指数是指他至多有h篇论文分别被引用了至少h次。
        h指数能够比较准确地反映一个人的学术成就。一个人的h指数越高，则表明他的论文影响力越大。
        例如，某人的h指数是20，这表示他已发表的论文中，每篇被引用了至少20次的论文总共有20篇。
        要确定一个人的h指数非常容易，到SCI网站，查出某个人发表的所有SCI论文，让其按被引次数从高到低排列，往下核对，直到某篇论文的序号大于该论文被引次数，那个序号减去1就是h指数。
        中国读者较为熟悉的霍金的h指数比较高，为62。
        生物学家当中h指数最高的为沃尔夫医学奖获得者、约翰斯·霍普金斯大学神经生物学家施奈德，高达191，
        其次为诺贝尔生理学或医学奖获得者、加州理工学院生物学家巴尔的摩，160。
        生物学家的h指数都偏高，表明h指数就像其他指标一样，不适合用于跨学科的比较。

        计算网络节点的H-index
        思路：考虑节点的度属性和邻居关系，即：
            一个节点的h-index为k，表示至少有k个邻居的度不小于k，有向网络仅考虑入度
        :return: dict
        """

        def countHindex(ls):
            if ls:
                ls.sort()  # 排序算法 最耗时的部分
                h = 1 if ls[-1] > 0 else 0
                small = ls[-1]
                for i in ls[-2::-1]:
                    if i == small and i > h:
                        h += 1
                    elif i > h:
                        h += 1
                        small = i
                    else:
                        break
                return h
            else:
                return 0

        if type == "in":
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应指定type=\"\"")
            else:
                degrees = cls.inDegree()
                neighbors = cls.inNeighbor()
        elif type == "out":
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应指定type=\"\"")
            else:
                degrees = cls.outDegree()
                neighbors = cls.outNeighbor()
        else:
            if cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "有向网络应指定type=\"in\"或者\"out\"")
            else:
                degrees = cls.degree()
                neighbors = cls.neighbor()

        try:
            h_index = {node: countHindex([degrees[i] for i in neighbors[node]]) for node in cls.network}
            return CommonUtils.orderDict(h_index, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def h2(cls, type=""):

        if type == "in":
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应指定type=\"\"")
            else:
                dhcs = cls.inStrength()
                nbrs = cls.inNeighbor()
        elif type == "out":
            if not cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "无向网络应指定type=\"\"")
            else:
                dhcs = cls.outStrength()
                nbrs = cls.outNeighbor()
        else:
            if cls.isDirected():
                raise BusinessException(EmBusinessError.NETWORK_MISMATCH_ERROR, "有向网络应指定type=\"in\"或者\"out\"")
            else:
                dhcs = cls.strength()
                nbrs = cls.neighbor()

        res = dict()
        try:
            for node in dhcs:
                dhcs = CommonUtils.orderDict(dhcs, index=1)
                neighbors = nbrs.get(node)
                neighbors = {node: neighbors.get(node) for node in dhcs if node in neighbors}
                if neighbors:
                    x = [nbr['weight'] for nbr in neighbors.values()]
                    y = [dhcs.get(nbr) for nbr in neighbors]
                    res[node] = cls._dhc(x, y)
                else:
                    res[node] = 0
            return CommonUtils.orderDict(res, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    # CI ---------------------------------------------------------------------------------------------------------
    @classmethod
    def CI(cls, isWeighted=False):
        """
        计算网络传播指标CI: collective influence

        The idea is to determine the nodes with the most influence on a network
        that are capable of collapsing or totally fragmenting the network when they are removed.
        After the first iteration, the CI value is calculated and sorted in descending order.
        The node with the highest CI value is removed and the calculation is repeated and CI sorted again
        to determine the node with the next highest CI value.
        The cycle continues until the network is completely fragmented with the least number of largest components.

        计算公式：CI(i)=(ki - 1) * sum([kj -1 for j in neighbor(i)])
        :return: CI
        """
        neighbors_info = cls.neighbor()
        CI_info = {node: 0 for node in cls.network}
        degree_info = cls.degree()

        try:
            for node in cls.network:
                if degree_info[node] > 1:
                    if isWeighted:
                        strength_info = cls.strength()
                        CI_info[node] = (degree_info[node] - 1) * sum(
                            [(strength_info[n] - neighbors_info[node].get(n)['weight']) if strength_info[n] > 1 else 0
                             for n in neighbors_info[node]])
                    else:
                        CI_info[node] = (degree_info[node] - 1) * sum(
                            [(degree_info[n] - 1) if degree_info[n] > 1 else 0 for n in neighbors_info[node]])
            return CommonUtils.orderDict(CI_info, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def CIIn(cls):
        neighbors_info = cls.inNeighbor()
        degree_info = cls.inDegree()
        CI_info = {node: 0 for node in cls.network}

        try:
            for node in cls.network:
                if degree_info[node] > 1:
                    CI_info[node] = (degree_info[node] - 1) * sum(
                        [(degree_info[n] - 1) if degree_info[n] > 1 else 0 for n in neighbors_info[node]])
            return CommonUtils.orderDict(CI_info, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def CIOut(cls):
        neighbors_info = cls.outNeighbor()
        degree_info = cls.inDegree()
        CI_info = {node: 0 for node in cls.network}

        try:
            for node in cls.network:
                if degree_info[node] > 1:
                    CI_info[node] = (degree_info[node] - 1) * sum(
                        [(degree_info[n] - 1) if degree_info[n] > 1 else 0 for n in neighbors_info[node]])
            return CommonUtils.orderDict(CI_info, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    # 核数 -------------------------------------------------------------------------------------------------------
    @classmethod
    def _coreNum(cls):
        """Returns the core number for each vertex.

        A k-core is a maximal subgraph that contains nodes of degree k or more.

        The core number of a node is the largest value k of a k-core containing
        that node.

        Parameters
        ----------
        G : NetworkX graph
           A graph or directed graph

        Returns
        -------
        core_number : dictionary
           A dictionary keyed by node to the core number.

        Raises
        ------
        NetworkXError
            The k-core is not implemented for graphs with cls loops
            or parallel edges.

        Notes
        -----
        Not implemented for graphs with parallel edges or cls loops.

        For directed graphs the node degree is defined to be the
        in-degree + out-degree.

        References
        ----------
        .. [1] An O(m) Algorithm for Cores Decomposition of Networks
           Vladimir Batagelj and Matjaz Zaversnik, 2003.
           https://arxiv.org/abs/cs.DS/0310049
        """
        try:
            return CommonUtils.orderDict(nx.core_number(cls.network), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def _inCoreNum(cls):

        degrees = dict(cls.inDegree())
        # Sort nodes by degree.
        nodes = sorted(degrees, key=degrees.get)
        bin_boundaries = [0]
        curr_degree = 0
        for i, v in enumerate(nodes):
            if degrees[v] > curr_degree:
                bin_boundaries.extend([i] * (degrees[v] - curr_degree))
                curr_degree = degrees[v]
        node_pos = {v: pos for pos, v in enumerate(nodes)}
        core = degrees
        in_nbrs = cls.outNeighbor()

        try:
            nbrs = {v: list(in_nbrs.get(v)) for v in in_nbrs}
            for v in nodes:
                for u in nbrs[v]:
                    if core[u] > core[v]:
                        nbrs[v].remove(u)
                        pos = node_pos[u]
                        bin_start = bin_boundaries[core[u]]
                        node_pos[u] = bin_start
                        node_pos[nodes[bin_start]] = pos
                        nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                        bin_boundaries[core[u]] += 1
                        core[u] -= 1
            return CommonUtils.orderDict(core, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def _outCoreNum(cls):
        degrees = dict(cls.outDegree())
        # Sort nodes by degree.
        nodes = sorted(degrees, key=degrees.get)
        bin_boundaries = [0]
        curr_degree = 0
        for i, v in enumerate(nodes):
            if degrees[v] > curr_degree:
                bin_boundaries.extend([i] * (degrees[v] - curr_degree))
                curr_degree = degrees[v]
        node_pos = {v: pos for pos, v in enumerate(nodes)}
        # The initial guess for the core number of a node is its degree.
        core = degrees
        out_nbrs = cls.inNeighbor()

        try:
            nbrs = {v: list(out_nbrs.get(v)) for v in out_nbrs}
            for v in nodes:
                for u in nbrs[v]:
                    if core[u] > core[v]:
                        nbrs[v].remove(u)
                        pos = node_pos[u]
                        bin_start = bin_boundaries[core[u]]
                        node_pos[u] = bin_start
                        node_pos[nodes[bin_start]] = pos
                        nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                        bin_boundaries[core[u]] += 1
                        core[u] -= 1
            return CommonUtils.orderDict(core, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def _dhc(cls, x, y):
        """
        作图法求解节点的dhc值
        :param x: edge weight list
        :param y: dhc list
        :return: dhc, float

        piecewise_function：分段函数
        """

        def piecewise_function(x, y, x0):
            """
            分段函数
            :param x: 交点x坐标，list
            :param y: 交点y坐标，list
            :param x0: 目标点x坐标，float
            :return: 目标点y坐标
            注：默认首个焦点为与y轴交点，且不在x中
            """
            for i in range(len(x)):
                if x0 <= x[i]:
                    return y[i]
            return 0

        if 0 in y:  # 邻居的dhc值为零时，舍去
            for index, item in enumerate(y):
                if item == 0:
                    y.pop(index)
                    x.pop(index)

        if not x:  # 邻居的dhc全为0时，返回0
            return 0

        x = [sum(x[:i]) for i in range(1, len(x) + 1)]

        left = 0
        right = x[-1] + 0.01
        left_value = piecewise_function(y=y, x=x, x0=left) - left
        right_value = piecewise_function(y=y, x=x, x0=right) - right
        EPS = right / 10000
        while right - left >= EPS:  # 二分法求零点
            center = (right + left) / 2
            center_value = piecewise_function(y=y, x=x, x0=center) - center
            if center_value == 0:
                return center
            elif left_value == 0:
                return left
            elif right_value == 0:
                return right
            elif left_value * center_value < 0:
                right = center
                right_value = piecewise_function(y=y, x=x, x0=right) - right
            elif right_value * center_value < 0:
                left = center
                left_value = piecewise_function(y=y, x=x, x0=left) - left
        # return round(center, 5)
        return center

    @classmethod
    def _weightedCoreNum(cls, nbrs, dhcs, ITER=10000):
        try:
            import random
            nodes = list(dhcs)
            for i in range(ITER):
                dhcs = CommonUtils.orderDict(dhcs, index=1)
                node = random.choice(nodes)
                neighbors = nbrs.get(node)
                neighbors = {_: neighbors.get(_) for _ in dhcs if _ in neighbors}

                if neighbors:
                    x = [nbr['weight'] for nbr in neighbors.values()]
                    y = [dhcs.get(nbr) for nbr in neighbors]
                    dhcs[node] = cls._dhc(x, y)
                else:
                    dhcs[node] = 0
            return CommonUtils.orderDict(dhcs, index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def core(cls, isWeighted=False):
        """
        计算无向网络核数
        :param weighted: bool
        :return: dict
        """
        try:
            if isWeighted:
                return CommonUtils.orderDict(cls._weightedCoreNum(cls.neighbor(), cls.strength()), index=1)
            else:
                return CommonUtils.orderDict(cls._coreNum(), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def inCore(cls, isWeighted=False):
        """
        计算有向网络入核数
        :param weighted: bool
        :return: dict
        """
        try:
            if isWeighted:
                return CommonUtils.orderDict(cls._weightedCoreNum(cls.inNeighbor(), cls.inStrength()), index=1)
            else:
                return CommonUtils.orderDict(cls._inCoreNum(), index=1)
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_CUSTOM_METRIC_ERROR, errMsg)

    @classmethod
    def outCore(cls, isWeighted=False):
        """
        计算有向网络出核数
        :param weighted: bool
        :return: dict
        """

        if isWeighted:
            return CommonUtils.orderDict(cls._weightedCoreNum(cls.outNeighbor(), cls.outStrength()), index=1)
        else:
            return CommonUtils.orderDict(cls._outCoreNum(), index=1)


class ComplexNetworkService(Network, NetworkBaseInfo, NetworkLibraryMetric, NetworkCustomMetric):
    def __init__(self, complexNetworkModel):
        self.setNetwork(complexNetworkModel)
