class Module(object):

    def __init__(self):
        self._parameters = None
        self._gradient = None

    def update_parameters(self, gradient_step=1e-3):
        # Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient

    def zero_grad(self):
        # Annule gradient
        pass

    def forward(self, X):
        # Calcule la passe forward
        pass

    def backward_update_gradient(self, X, delta):
        # Met a jour la valeur du gradient
        pass

    def backward_delta(self, X, delta):
        # Calcul la derivee de l'erreur
        pass
