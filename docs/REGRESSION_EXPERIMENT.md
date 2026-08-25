# Closing the loop on a class-dependent regression

The learned candidate beat the kinematic ensemble by a wide margin in aggregate
and still made 98 agents worse. Characterising those agents produced a
hypothesis, the hypothesis produced a fix, and the same tooling that found the
defect was used to decide whether the fix worked.

This records that cycle, including the parts that did not come out cleanly.

## The defect

Regression rates against the kinematic ensemble, by object type:

| Type | Agents | Regressed | Share of training data |
|---|---:|---:|---:|
| Vehicle | 1,055 | 6.2% | 83.6% |
| Pedestrian | 126 | **23.0%** | 14.8% |
| Cyclist | 22 | **18.2%** | 1.6% |

The classes the model saw least were the ones it degraded most. None of this is
visible in the aggregate minADE of 2.008 m.

## The diagnosis

Data volume was the obvious explanation and was not the whole one. Reviewing the
feature layout showed that **neighbouring agents carried an object-type
indicator while the target did not**. An embedding for the target's own class
was declared in the model and never wired into the forward pass.

The network therefore had to infer whether it was predicting a car or a
pedestrian from motion alone, with a training population dominated by vehicles.
Defaulting to vehicle-like behaviour is the expected consequence.

## The candidates

Object type is recorded during consolidation, so correcting this required no
reprocessing of the feature archives.

| Run | Change |
|---|---|
| A | Original model |
| B | Target object type supplied to the network |
| C | B, plus class-balanced sampling at damping 0.5 |

Each trained for 30 epochs on the same 67,574 examples, and each was scored on
the same 1,203 designated targets of the verified validation shard.

## Aggregate result

| Run | minADE | minFDE | miss rate |
|---|---:|---:|---:|
| A | 2.008 m | 4.638 m | 0.657 |
| B | 1.979 m | 4.568 m | 0.660 |
| C | 2.011 m | 4.698 m | 0.661 |

Essentially unchanged. Run through the regression gate, **B does not pass
against A**: the paired 95% interval on minADE is [-0.073, +0.017], which
includes zero, and agent outcomes are close to a coin flip at 632 improved
against 571 regressed.

The gate is correct to refuse. There is no aggregate improvement to certify.

## Subgroup result

The claim under test was never about the aggregate. Paired agent-level bootstrap
on the classes the fix targeted, 10,000 resamples, fixed seed:

| Class | Run | minADE change | 95% interval | Supported |
|---|---|---:|---|---|
| Pedestrian | B | -0.112 m | [-0.176, -0.049] | **yes** |
| Pedestrian | C | -0.100 m | [-0.166, -0.038] | **yes** |
| Cyclist | B | -0.153 m | [-0.691, +0.429] | no |
| Cyclist | C | -0.317 m | [-0.966, +0.350] | no |
| Vehicle | B | -0.017 m | [-0.066, +0.033] | no |
| Vehicle | C | +0.022 m | [-0.034, +0.078] | no |

And on the defect itself, the rate at which each model loses to the kinematic
ensemble:

| Class | Run | Rate change | 95% interval | Supported |
|---|---|---:|---|---|
| Pedestrian | B | -0.071 | [-0.143, +0.000] | no, marginally |
| Pedestrian | C | **-0.087** | [-0.159, -0.024] | **yes** |
| Cyclist | B | -0.045 | [-0.136, +0.000] | no |
| Cyclist | C | -0.045 | [-0.136, +0.000] | no |

Observed regression rates fell from 23.0% to 15.9% (B) and 14.3% (C) for
pedestrians, and from 18.2% to 13.6% for cyclists under both.

## Conclusions

**The fix worked, on the population it was aimed at.** Both candidates improve
pedestrian accuracy with statistical support. Only C significantly reduces the
rate at which pedestrians lose to the kinematic baseline, which was the defect.

**Neither fix measurably harmed vehicles.** The balancing cost that C was
expected to pay is not detectable at this sample size; its interval includes
zero.

**The cyclist result cannot be claimed.** Twenty-two agents cannot support a
conclusion, and every cyclist interval straddles zero. The observed improvement
is encouraging and unproven, and reporting it as a finding would be wrong.

**C is the better candidate for the diagnosed defect**, despite being marginally
worse in aggregate. That is only a defensible conclusion because the aggregate
was never the target.

## Why this is the point

The aggregate moved 1.4%. The pedestrian regression rate moved 31%.

A leaderboard number would have called this change worthless, and the regression
gate agreed — correctly, because no aggregate improvement exists. Only the
class-level evidence showed that a real, targeted defect had been reduced with
statistical support.

That is the argument for building the evidence layer, demonstrated rather than
asserted: the harness found a defect an aggregate concealed, the diagnosis
pointed at a specific cause, and the same harness both confirmed the targeted
improvement and refused to certify the improvement that was not there.
