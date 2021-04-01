import networkx as nx

from error.BusinessException import BusinessException
from error.EmBusinessError import EmBusinessError


class ComplexNetworkModel:
    network = None

    # 生成网络
    @classmethod
    def generateNetwork(cls, edges, isWeighted=False):
        """生成无向网络"""
        if not edges or len(edges) == 0:
            return
        elif isWeighted and len(edges[0]) != 3:
            raise BusinessException(EmBusinessError.NETWORK_GENERATE_FAIL, "无向含权网络的边：(head, tail, weight)，item长度至少为3")
        elif not isWeighted and len(edges[0]) < 2:
            raise BusinessException(EmBusinessError.NETWORK_GENERATE_FAIL, "无向无权网络的边：(head, tail)，item长度至少为3")

        cls.network = nx.Graph()
        try:
            if isWeighted:
                for edge in edges:
                    cls.network.add_edge(edge[0], edge[1], weight=float(edge[2]))
            else:
                for edge in edges:
                    cls.network.add_edge(edge[0], edge[1])
            return cls
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_GENERATE_FAIL, errMsg)

    @classmethod
    def generateDirectNetwork(cls, edges, isWeighted=False):
        """生成有向网络"""
        if not edges or len(edges) == 0:
            return
        elif isWeighted and len(edges[0]) != 3:
            raise BusinessException(EmBusinessError.NETWORK_GENERATE_FAIL, "有向含权网络的边：(head, tail, weight)，item长度至少为3")
        elif not isWeighted and len(edges[0]) < 2:
            raise BusinessException(EmBusinessError.NETWORK_GENERATE_FAIL, "有向无权网络的边：(head, tail)，item长度至少为3")

        cls.network = nx.DiGraph()
        try:
            if isWeighted:
                for edge in edges:
                    cls.network.add_edge(edge[0], edge[1], weight=float(edge[2]))
            else:
                for edge in edges:
                    cls.network.add_edge(edge[0], edge[1])
            return cls
        except Exception as errMsg:
            raise BusinessException(EmBusinessError.NETWORK_GENERATE_FAIL, errMsg)

    @classmethod
    def randomRegularGraph(cls, d=3, n=20):
        r"""Returns a random $d$-regular graph on $n$ nodes.

        The resulting graph has no self-loops or parallel edges.

        Parameters
        ----------
        d : int
          The degree of each node.
        n : integer
          The number of nodes. The value of $n \times d$ must be even.
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Notes
        -----
        The nodes are numbered from $0$ to $n - 1$.

        Kim and Vu's paper [2]_ shows that this algorithm samples in an
        asymptotically uniform way from the space of random graphs when
        $d = O(n^{1 / 3 - \epsilon})$.

        Raises
        ------

        NetworkXError
            If $n \times d$ is odd or $d$ is greater than or equal to $n$.

        References
        ----------
        .. [1] A. Steger and N. Wormald,
           Generating random regular graphs quickly,
           Probability and Computing 8 (1999), 377-396, 1999.
           http://citeseer.ist.psu.edu/steger99generating.html

        .. [2] Jeong Han Kim and Van H. Vu,
           Generating random regular graphs,
           Proceedings of the thirty-fifth ACM symposium on Theory of computing,
           San Diego, CA, USA, pp 213--222, 2003.
           http://portal.acm.org/citation.cfm?id=780542.780576
        """

        cls.network = nx.random_graphs.random_regular_graph(d, n)
        return cls

    @classmethod
    def erdosRenyiGraph(cls, n=20, p=0.2, **kwargs):
        """Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
        or a binomial graph.

        The $G_{n,p}$ model chooses each of the possible edges with probability $p$.

        The functions :func:`binomial_graph` and :func:`erdos_renyi_graph` are
        aliases of this function.

        Parameters
        ----------
        n : int
            The number of nodes.
        p : float
            Probability for edge creation.
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.
        directed : bool, optional (default=False)
            If True, this function returns a directed graph.

        See Also
        --------
        fast_gnp_random_graph

        Notes
        -----
        This algorithm [2]_ runs in $O(n^2)$ time.  For sparse graphs (that is, for
        small values of $p$), :func:`fast_gnp_random_graph` is a faster algorithm.

        References
        ----------
        .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
        """
        cls.network = nx.random_graphs.erdos_renyi_graph(n, p, seed=kwargs.get('seed'),
                                                         directed=kwargs.get('directed', False))
        return cls

    @classmethod
    def wattsStrogatz(cls, n=20, k=4, p=0.3):
        """Returns a Watts–Strogatz small-world graph.

        Parameters
        ----------
        n : int
            The number of nodes
        k : int
            Each node is joined with its `k` nearest neighbors in a ring
            topology.
        p : float
            The probability of rewiring each edge
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        See Also
        --------
        newman_watts_strogatz_graph()
        connected_watts_strogatz_graph()

        Notes
        -----
        First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
        to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
        Then shortcuts are created by replacing some edges as follows: for each
        edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
        with probability $p$ replace it with a new edge $(u, w)$ with uniformly
        random choice of existing node $w$.

        In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
        does not increase the number of edges. The rewired graph is not guaranteed
        to be connected as in :func:`connected_watts_strogatz_graph`.

        References
        ----------
        .. [1] Duncan J. Watts and Steven H. Strogatz,
           Collective dynamics of small-world networks,
           Nature, 393, pp. 440--442, 1998.
        """
        # 基于WS小世界模型
        cls.network = nx.random_graphs.watts_strogatz_graph(n, k, p)
        return cls

    @classmethod
    def barabásiAlbert(cls, n=20, m=2):
        """Returns a random graph according to the Barabási–Albert preferential
        attachment model.

        A graph of $n$ nodes is grown by attaching new nodes each with $m$
        edges that are preferentially attached to existing nodes with high degree.

        Parameters
        ----------
        n : int
            Number of nodes
        m : int
            Number of edges to attach from a new node to existing nodes
        seed : integer, random_state, or None (default)
            Indicator of random number generation state.
            See :ref:`Randomness<randomness>`.

        Returns
        -------
        G : Graph

        Raises
        ------
        NetworkXError
            If `m` does not satisfy ``1 <= m < n``.

        References
        ----------
        .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
           random networks", Science 286, pp 509-512, 1999.
        """
        cls.network = nx.random_graphs.barabasi_albert_graph(n, m)
        return cls


if __name__ == "__main__":
    a = ComplexNetworkModel()
    b = ComplexNetworkModel.barabásiAlbert()
    c = ComplexNetworkModel.wattsStrogatz()
    d = ComplexNetworkModel.randomRegularGraph()
    e = ComplexNetworkModel.generateNetwork(edges=[(1, 2), (3, 4)])
