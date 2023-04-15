import torch
import torch.nn as nn
from models import snn, functional as sf


class Network(nn.Module):

    def __init__(self, device='cuda'):
        super(Network, self).__init__()
        self.device = device

        self.conv1 = snn.Convolution(
            in_channels=2, out_channels=32, kernel_size=5)
        self.conv1_threshold = 10
        self.conv1_kwinners = 5
        self.conv1_inhibition_rad = 2

        self.conv2 = snn.Convolution(
            in_channels=32, out_channels=150, kernel_size=2)
        self.conv2_threshold = 1
        self.conv2_kwinners = 8
        self.conv2_inhibition_rad = 1

        self.stdp1 = snn.STDP(conv_layer=self.conv1)
        self.stdp2 = snn.STDP(conv_layer=self.conv2)

        self.max_ap = torch.Tensor([0.15]).to(self.device)

        self.ctx = {"input_spikes": None, "potentials": None,
                    "output_spikes": None, "winners": None}
        self.spk_cnt = 0

    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, layer_idx=None):
        input = sf.pad(input.float(), (2, 2, 2, 2))

        if self.training:
            potentials = self.conv1(input)
            spk, pot = sf.fire(
                potentials=potentials, threshold=self.conv1_threshold,
                return_thresholded_potentials=True
            )

            if layer_idx == 1:
                self.spk_cnt += 1
                if self.spk_cnt >= 500:
                    self.spk_cnt = 0
                    ap = self.stdp1.learning_rate[0]\
                        .clone().detach()\
                        .to(self.device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_learning_rate(ap, an)

                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(
                    pot, self.conv1_kwinners, self.conv1_inhibition_rad, spk)
                self.save_data(input, pot, spk, winners)

            else:
                spk_pooling = sf.pooling(spk, 2, 2, 1)
                spk_in = sf.pad(spk_pooling, (1, 1, 1, 1))
                spk_in = sf.pointwise_inhibition(spk_in)

                potentials = self.conv2(spk_in)
                spk, pot = sf.fire(potentials, self.conv2_threshold, True)
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(
                    pot, self.conv2_kwinners, self.conv2_inhibition_rad, spk)

                self.save_data(spk_in, pot, spk, winners)

        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_threshold, True)
            pooling = sf.pooling(spk, 2, 2, 1)
            padded = sf.pad(pooling, (1, 1, 1, 1))

            pot = self.conv2(padded)
            spk, pot = sf.fire(pot, self.conv2_threshold, True)
            # pooled_spk, _ = torch.max(spk.reshape(spk.size(1), -1), dim=1)
            # spk_out = pooled_spk.view(1, spk.size(1))
            spk_out = sf.pooling(spk, 2, 2, 1)
            return spk_out

    def stdp(self, layer_idx):
        if layer_idx == 1:
            stdpn = self.stdp1
        else:
            stdpn = self.stdp2
        stdpn(self.ctx["input_spikes"],
              self.ctx["output_spikes"], self.ctx["winners"])
