# Judging forecasts by behaviour, not distance

Displacement error cannot say whether being wrong mattered.

A forecast can be five metres out harmlessly — a parallel offset along an empty
straight road — and two metres out catastrophically, by predicting a vehicle
stops at a line it actually crosses. minADE reports the first as worse.

`verify-predictions` asks a different question. It rebuilds the scenario as the
model believes it will unfold, runs the same declarative requirements used on
recorded trajectories, and compares the verdicts.

## What comes back

Not a distance. A behavioural claim, in one of four states:

| Verdict | Meaning |
|---|---|
| Correct | Forecast and record agree on whether the rule held |
| **False alarm** | The forecast implies a violation the record does not contain |
| **Missed violation** | The record contains a violation the forecast does not |
| Not comparable | The requirement did not apply to one of the two |

The two middle rows are what review exists for, and they are not symmetric. A
false alarm makes a consumer brake for nothing. A missed violation lets it
proceed into something real.

## Method

Both views are sampled at the prediction timestamps. A recorded track carries
states every 0.1 s while a submission carries sixteen points at 0.5 s spacing;
comparing those directly would attribute the difference in sampling to the
model. Resampling the record to the forecast's own timestamps removes that.

Only the target agent's future is replaced. Every other agent keeps its recorded
trajectory, so the question asked is precise: *if the model is right about this
agent, does a rule break?*

The **most confident** mode is judged by default, because that is the forecast a
consumer would act on. Judging the closest mode instead would flatter the model
by selecting a trajectory with hindsight — which is exactly what minADE does,
and exactly why minADE is not a behavioural measure.

## Result

Three candidates over the 1,203 designated targets of the verified validation
shard, against speed, acceleration and jerk requirements:

| Candidate | minADE | Behavioural accuracy | False alarms | Missed violations |
|---|---:|---:|---:|---:|
| Constant velocity | 9.633 m | **92.9%** | **12** | 246 |
| Kinematic ensemble | 7.728 m | **92.9%** | **12** | 246 |
| Learned model | **2.008 m** | 65.0% | 1,167 | **97** |

**The ranking inverts.** The candidate that is 4.8x more accurate spatially is
the worst behaviourally, and the two candidates that lose badly on displacement
produce the most plausible driving.

Constant velocity and the kinematic ensemble score identically because the
ensemble's most confident mode is constant velocity for all 1,203 agents, so the
same trajectory is being judged twice.

### Why the learned model loses

Its predicted trajectories are not kinematically plausible:

| Peak acceleration over the horizon | Recorded | Learned model |
|---|---:|---:|
| Median | 1.18 m/s² | **2.19 m/s²** |
| Mean | 1.33 m/s² | 2.64 m/s² |
| Agents exceeding 3 m/s² | 5.7% | **32.7%** |

Real driving exceeds 3 m/s² in one agent in eighteen. The model's forecasts do
it in one in three. Jerk is worse still: 36.8% of its jerk verdicts are correct,
against 92.8% for constant velocity.

The trajectories wobble around the true path closely enough to score well on
displacement while implying harsh braking and acceleration that never happened.
A planner consuming them would react to events that do not exist.

Constant velocity cannot produce that failure, because a straight line at fixed
speed has no acceleration and no jerk by construction. Its plausibility is not a
virtue the model earned; it is a property of the assumption.

### Why the learned model still wins where it counts

The two families fail in opposite directions.

| | False alarms | Missed violations |
|---|---:|---:|
| Constant velocity | 12 | 246 |
| Learned model | 1,167 | **97** |

The physics baselines never cry wolf and miss 246 real violations, because a
constant-velocity assumption structurally cannot anticipate a vehicle braking
hard. The learned model cries wolf constantly and misses 97 — **2.5x fewer**.

Which failure is worse depends on the consumer, and this tool does not decide
that. It reports both, which no aggregate does.

## Prior work

None of the ideas here are new, and the literature says so more thoroughly than
this document does.

The argument that displacement metrics are not safety-relevant is established.
[Beyond ADE and FDE](https://arxiv.org/abs/2510.10086) sets out an evaluation
framework for safety-critical prediction on exactly this basis, and
[What Truly Matters in Trajectory Prediction for Autonomous Driving?](https://arxiv.org/abs/2306.15136)
makes the related case that dataset metrics do not track driving performance.

The kinematic implausibility measured above is also known.
[Physically Feasible Vehicle Trajectory Prediction](https://arxiv.org/abs/2104.14679)
studies predictions that violate vehicle dynamics, and **jerk violation rate**
is an established metric with published comfort thresholds in the range of
roughly 0.3 to 0.9 m/s³ — considerably stricter than the 5 m/s³ used here, which
was chosen to exercise the machinery rather than to reflect ride comfort.

What this repository contributes is not the idea but an implementation: one
declarative requirement file that runs against both recorded and predicted
trajectories, reporting paired agreement with the record rather than a violation
rate in isolation. No prior art was found for that specific integration, on a
handful of searches rather than a literature review — a weak claim, and stated
weakly on purpose.

## Interpretation boundary

Requirement thresholds here are chosen to exercise the machinery. A behavioural
disagreement is evidence about this requirement, this threshold and this
recording. It is not a collision probability, not a safety determination, and
not a claim about any production system.

Smoothing the model's outputs would improve every number in this document
without making its predictions any more correct. That is worth stating plainly:
this measure is gameable in a way displacement is not, and it is meant to sit
beside displacement rather than replace it.

## Usage

```bash
verify-predictions candidate.binproto data/raw/SHARD \
  examples/predicted_behaviour_requirements.json \
  --json-report reports/generated/behaviour.json
```

Requirements are templates: a `subject_agent_id` of `@target` binds to each
predicted agent in turn, so one file covers a whole shard. The command exits `1`
when any missed violation is found.
