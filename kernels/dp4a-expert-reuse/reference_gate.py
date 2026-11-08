"""Broad numerical sanity gate, not a model-quality equivalence criterion.

Fixed before further GPU trials: error <= 0.35 and <= 1.25 * validated
CUTLASS error + 0.01. These intentionally broad limits catch broken contracts;
passing does not establish equal arithmetic or quality. Reject zero collapse
independently of error thresholds, and require the comparator itself to pass.
"""
import math
ABS_REL_CEILING = 0.35
BASELINE_ERROR_FACTOR = 1.25
BASELINE_ERROR_MARGIN = 0.01

def check(errors, output_norms, reference_norm):
 if not math.isfinite(reference_norm) or reference_norm < 0:
  raise ValueError('invalid independent reference norm')
 base = errors['current_scalar_cutlass']['rel_l2']
 if not math.isfinite(base) or base > ABS_REL_CEILING:
  raise ValueError('CUTLASS reference error exceeds broad sanity ceiling')
 limit = min(ABS_REL_CEILING, BASELINE_ERROR_FACTOR * base + BASELINE_ERROR_MARGIN)
 failures = []
 for name, metrics in errors.items():
  norm = output_norms[name]
  if not metrics['finite'] or not math.isfinite(norm) or norm < 0 or not math.isfinite(metrics['rel_l2']):
   failures.append(name + ': nonfinite or invalid output/error')
  elif reference_norm > 0 and norm == 0:
   failures.append(name + ': zero output for nonzero independent reference')
  elif metrics['rel_l2'] > limit:
   failures.append(name + ': independent reference error exceeds sanity limit')
 if failures:
  raise ValueError('; '.join(failures))
 return dict(status='broad_sanity_only', absolute_ceiling=ABS_REL_CEILING,
             baseline_factor=BASELINE_ERROR_FACTOR, baseline_margin=BASELINE_ERROR_MARGIN,
             effective_limit=limit, quality_equivalence=False)
