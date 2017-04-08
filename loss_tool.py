from visdom_helper.visdom_helper import Dashboard
import torch

class Loss_tool(object):
    """
    Helper class to help visualize the loss change.
    """

    def __init__(self, name = None, window_sizes = [30]):
        """
        Initialization for the loss tool.

        Args: name: loss name, the default is none
        window_sizes: the smooth window sizes (default is [30], should be a list)
        """
        # raw loss value
        self.loss_raw = []

        # initialize the dashboard
        self.loss_dict = {}
        for window_size in window_sizes:
            self.loss_dict[window_size] = {}
            self.loss_dict[window_size]['loss'] = []
            self.loss_dict[window_size]['count'] = 0
            self.loss_dict[window_size]['x'] = []
            self.loss_dict[window_size]['name'] = "loss-" + str(window_size)
            if name:
                self.loss_dict[window_size]['name'] = name + "-" + self.loss_dict[window_size]['name']

        # init the dashboard
        self.dashboard = Dashboard("loss_env")

    def append_loss(self, loss):
        """
        Add a new loss value

        Args:
        loss: the loss value
        """

        if type(loss) == torch.Tensor:
            self.loss_raw += loss.tolist()
        else:
            raise ValueError("type: {0} is not support yet, only support torch tensor".format(type(loss)))

        # update loss for all different window size
        self._update_loss_array()

        # plot the loss
        self.plot_all()

    def _update_loss_array(self):
        """
        Update the loss array
        """

        for size in self.loss_dict:
            if len(self.loss_raw) > size:
                self.loss_dict[size]['count'] += 1
                self.loss_dict[size]['x'].append(self.loss_dict[size]['count'])
                self.loss_dict[size]['loss'].append(
                    sum(self.loss_raw[-size:]) / size
                )

    def plot_all(self):
        """
        Plot all the curves
        """
        for size in self.loss_dict:
            name = self.loss_dict[size]['name']
            loss_values = self.loss_dict[size]['loss']
            x = self.loss_dict[size]['x']
            if len(loss_values) > 0:
                self.dashboard.plot(name, "line", X=torch.FloatTensor(x), Y=torch.FloatTensor(loss_values))
