"""Isolated experiment: compile original native QuantFP8 into caller buffers.

Not selected by the GDN overlay. GPU exactness/codegen/timing remain required.
No torch/vLLM imports or device activity occur by importing this module.
"""


def make_raw_quant_out(quantizer):
    def quant_out(x, q_out, scale_out):
        # Call the installed raw method, not its already compiled wrapper.
        q, scale = quantizer.forward_native(x, None, None)
        q_out.copy_(q)
        scale_out.copy_(scale)
    return quant_out
