# Workspace geometry by token band

Experimental integration hypothesis. No candidate GPU correctness or timing result yet.

The installed FlashInfer selector supports M16/32/64/128, but the graph wrapper
allocates one dynamic workspace from `max_num_tokens * top_k`. When that workspace
is passed explicitly, `launch_sm120_moe` retains its stored `tile_m`. Current
scalar uses a large maximum and therefore M128 even for smaller dynamic calls.
Saved actual L19/M2048 metadata confirms M128, while the functional selector for
the same token count would choose M64. A selector's existence does not establish
which tile runs in vLLM.

Keep the current CUTLASS dispatch below M1024. Prepare separate, immutable graph
wrappers for M1024–1535 (capacity1535, M32), M1536–3071 (capacity3071, M64), and
M3072+ (existing maximum, M128). Share scratch only within the existing execution
owner and band, retaining distinct layer weight views. This avoids changing a
workspace pointer under previously captured graphs. Preparation and extra memory
must be counted. The experimental flag must participate in vLLM compile factors.

First compare exact prepared operands and changing-input/route graph replay at
M1024/M2048, then complete expert-call timing against the current M128 wrapper.
Include an unchanged M4096 control and exercise band transitions. This is a
different experiment from changing task order, replacing native kernels, or
changing the router's expert choices. A smaller tile may lose through additional
weight loads or task overhead despite reducing padding.

`analyze_routes.py` reads saved real-route histograms and reports tile count and
padding for each candidate. It performs no GPU work and predicts no speedup.
