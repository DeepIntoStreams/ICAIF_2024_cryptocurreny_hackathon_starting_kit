from collections import defaultdict
import time


class BaseTrainer:
    def __init__(self, batch_size, G, G_optimizer, n_gradient_steps, foo=lambda x: x):
        self.batch_size = batch_size

        self.G = G
        self.G_optimizer = G_optimizer
        self.n_gradient_steps = n_gradient_steps

        self.losses_history = defaultdict(list)

        self.foo = foo

        self.init_time = time.time()

        #self.best_G = copy.deepcopy(G.state_dict())
        self.best_G_loss = None
        #self.best_G = copy.deepcopy(self.G.state_dict())
        self.config = None

    def save_model_dict(self):
        raise NotImplementedError('Model saving not implemented!')

    def toggle_grad(self, model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)