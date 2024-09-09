"""
This file contains the standard configuration of the NEAT algorithm
"""

ConnectInputs: bool = True                      # Pre connect each input/bias neuron to an output neuron
BiasNeurons: bool = True                        # Allows bias neurons
DefaultActivationFunction: str = "sigmoid"      # Default activation function for hidden layers

MutationChances: dict[str, float] = {"weight": 0,
                                     "remove_link": 50,
                                     "add_link": 50,
                                     "remove_node": 0,
                                     "add_node": 0,
                                     "activation": 0}

WeightPerturbingAmount = 0.3                    # How much can a weight be perturbed at most on a weight mutation
