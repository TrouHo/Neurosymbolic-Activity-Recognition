"""06 Models: sizes of the two networks, counted analytically from the layer definitions in
emob/models (no torch needed); tests/test_inspection.py checks the counts against the real models
where torch is installed.
"""
from __future__ import annotations

from emob import config as C
from inspection_data.common import Context, Registry

IN_CH = 17          # 13 orientation-invariant channels + 4 availability flags
HEAD_UNITS = sum(C.HEADS.values())


def conv1d(cin, cout, k):
    return cin * cout * k + cout


def lstm_layer(inp, hid, bidirectional=True):
    one = 4 * hid * (inp + hid) + 2 * 4 * hid           # PyTorch: W_ih, W_hh, b_ih, b_hh
    return one * (2 if bidirectional else 1)


def tcn_params(ch=64, layers=8):
    return conv1d(IN_CH, ch, 5) + layers * (conv1d(ch, ch, 3) + conv1d(ch, ch, 1)) + conv1d(ch, HEAD_UNITS, 1)


def deepconvlstm_params(conv_ch=64, kernel=5, conv_layers=4, units=128, out_ch=64):
    convs = (1 * conv_ch * kernel + conv_ch) + (conv_layers - 1) * (conv_ch * conv_ch * kernel + conv_ch)
    lstm = lstm_layer(conv_ch * IN_CH, units) + lstm_layer(2 * units, units)
    return convs + lstm + (2 * units * out_ch + out_ch) + conv1d(out_ch, HEAD_UNITS, 1)


def run(ctx: Context, reg: Registry):
    reg.section = "06 Models"
    reg.add("model.tcn.params", tcn_params(), "parameters", "code", "TCN with five heads (emob/models/tcn.py)")
    reg.add("model.deepconvlstm.params", deepconvlstm_params(), "parameters", "code", "DeepConvLSTM with five heads (emob/models/deepconvlstm.py)")
    reg.add("model.ratio", deepconvlstm_params() / tcn_params(), "-", "code", "DeepConvLSTM / TCN parameters")
    rf = 1 + 2 * (2 ** 8 - 1)
    reg.add("model.tcn.receptive_field_steps", rf, "steps", "code", "TCN receptive field: 1 + 2 (2^8 - 1) for kernel 3, dilations 1..128")
    reg.add("model.tcn.receptive_field_s", rf * C.STEP / C.FS, "s", "code", "same in seconds")
    reg.add("model.input_channels", IN_CH, "channels", "code", "phone 4 + hand 3 + waist 3 + leg 3 + 4 availability flags")
