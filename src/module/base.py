from collections import defaultdict


class Module(object):
    def __init__(self):
        self._parameters = defaultdict(None)
        self._gradient = defaultdict(None)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def update_parameters(self, gradient_step=1e-3):
        """
        Mets à jour les paramètres en utilisant le gradient qui a été calculé et le pas de gradient_step.
        """
        raise NotImplementedError()

    def zero_grad(self):
        """
        Annule le gradient
        """
        raise NotImplementedError()

    def forward(self, X):
        """
        Calcule la passe forward
        """
        raise NotImplementedError()

    def backward_update_gradient(self, X, delta):
        """
        Mets à jour la valeur du gradient
        """
        raise NotImplementedError()

    def backward_delta(self, X, delta):
        """
        Calcule la dérivée de l'erreur par rapport aux entrées.
        """
        raise NotImplementedError()
