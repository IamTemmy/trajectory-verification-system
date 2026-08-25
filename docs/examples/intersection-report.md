# Trajectory Validation Report — `synthetic-intersection-001`

**Overall result:** FAIL

- Tracks: 5
- Requirements passed: 2/3 applicable (3 total)

## Requirement summary

| Requirement | Result | Confidence | Evaluated | Failed | Observed range |
|---|---:|---:|---:|---:|---:|
| `INTERSECTION_SEPARATION_001` | FAIL | HIGH | 91 | 5 | 3.97–46.97 m |
| `INTERSECTION_VRU_SEPARATION_002` | PASS | HIGH | 91 | 0 | 6.17–31.61 m |
| `INTERSECTION_SPEED_003` | PASS | HIGH | 90 | 0 | 6.4–6.4 m/s |

## Data-quality and applicability annotations

- **INFO — `PARTIAL_REPORTED_VELOCITY`:** Reported velocity is incomplete; position-derived kinematics remain available.
- **INFO — `NO_MAP_FEATURES`:** No map features are attached; map-aware requirements are not applicable.

## `INTERSECTION_SEPARATION_001` — FAIL

Maintain at least 5 m between the through vehicle and the turning vehicle

Predicate: `separation greater_than_or_equal 5 m`.

Resolved agents: subject `through_vehicle`, counterpart `turning_vehicle`.

Evidence confidence: **HIGH** — At least five samples were evaluated with no blocking quality warning.

### Failure evidence

- separation was below the 5 m threshold from 5.4 s to 5.8 s; the worst value was 3.97 m (1.03 m beyond the limit).

### Threshold sensitivity

| Threshold | Result | Failed samples | Failed fraction |
|---:|---:|---:|---:|
| 4.5 m | FAIL | 4 | 4.4% |
| 5 m | FAIL | 5 | 5.5% |
| 5.5 m | FAIL | 7 | 7.7% |

## `INTERSECTION_VRU_SEPARATION_002` — PASS

Maintain at least 3 m between the through vehicle and the pedestrian

Predicate: `separation greater_than_or_equal 3 m`.

Resolved agents: subject `through_vehicle`, counterpart `pedestrian`.

Evidence confidence: **HIGH** — At least five samples were evaluated with no blocking quality warning.

No failed samples were observed.

### Threshold sensitivity

| Threshold | Result | Failed samples | Failed fraction |
|---:|---:|---:|---:|
| 2.7 m | PASS | 0 | 0.0% |
| 3 m | PASS | 0 | 0.0% |
| 3.3 m | PASS | 0 | 0.0% |

## `INTERSECTION_SPEED_003` — PASS

Keep the through vehicle at or below the 15 m/s urban limit

Predicate: `speed less_than_or_equal 15 m/s`.

Resolved agents: subject `through_vehicle`.

Evidence confidence: **HIGH** — At least five samples were evaluated with no blocking quality warning.

No failed samples were observed.

### Threshold sensitivity

| Threshold | Result | Failed samples | Failed fraction |
|---:|---:|---:|---:|
| 13.5 m/s | PASS | 0 | 0.0% |
| 15 m/s | PASS | 0 | 0.0% |
| 16.5 m/s | PASS | 0 | 0.0% |

## Interpretation boundary

A project-defined threshold failure is evidence about this requirement and trajectory record. It is not, by itself, proof of unsafe real-world operation or a claim about the production Waymo Driver.
