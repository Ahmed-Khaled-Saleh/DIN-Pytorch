
import networkx as nx
import matplotlib.pyplot as plt

class Topolgy:


      def __init__(self):
          self.n_agent = 80

      def generate_graph(self, graph_type='binomial', params=None):
          '''Generate connected connectivity graph according to the params.'''


          def erdos(params):
              if params < 2 / (self.n_agent - 1):
                  return
              G = None
              while G is None or nx.is_connected(G) is False:
                  return nx.erdos_renyi_graph(self.n_agent, params)


          graph_types = {
              'expander': nx.paley_graph(self.n_agent).to_undirected(),
              # 'grid': nx.grid_2d_graph(params, params),
              'cycle': nx.cycle_graph(self.n_agent),
              'path': nx.path_graph(self.n_agent),
              'star': nx.star_graph(self.n_agent - 1),
              'binomial': erdos(params),
              'complete': nx.complete_graph(self.n_agent),
              'line': nx.path_graph(self.n_agent),
              'geometric': nx.random_geometric_graph(self.n_agent, params),
              }


          if graph_type in graph_types:
              G = graph_types[graph_type]
          else:
              return

          self.n_edges = G.number_of_edges()
          self.G = G

    #   def plot_graph(self):
    #       '''Plot the generated connectivity graph.'''

    #       plt
    #       plt.figure()
    #       nx.draw(self.G)


