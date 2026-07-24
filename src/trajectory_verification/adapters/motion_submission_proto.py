"""Minimal wire-compatible classes for WOMD motion submissions."""

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _field(message, name, number, field_type, *, label=1, type_name=None, packed=False):
    field = message.field.add()
    field.name, field.number, field.label, field.type = name, number, label, field_type
    if type_name:
        field.type_name = type_name
    if packed:
        field.options.packed = True


def _build_classes():
    file = descriptor_pb2.FileDescriptorProto(
        name="trajectory_verification/motion_submission_subset.proto",
        package="waymo.open_dataset",
        syntax="proto2",
    )
    trajectory = file.message_type.add(); trajectory.name = "Trajectory"
    _field(trajectory, "center_x", 2, 2, label=3, packed=True)
    _field(trajectory, "center_y", 3, 2, label=3, packed=True)

    scored = file.message_type.add(); scored.name = "ScoredTrajectory"
    _field(scored, "trajectory", 1, 11, type_name=".waymo.open_dataset.Trajectory")
    _field(scored, "confidence", 2, 2)

    single = file.message_type.add(); single.name = "SingleObjectPrediction"
    _field(single, "object_id", 1, 5)
    _field(single, "trajectories", 2, 11, label=3,
           type_name=".waymo.open_dataset.ScoredTrajectory")

    prediction_set = file.message_type.add(); prediction_set.name = "PredictionSet"
    _field(prediction_set, "predictions", 1, 11, label=3,
           type_name=".waymo.open_dataset.SingleObjectPrediction")

    scenario = file.message_type.add(); scenario.name = "ChallengeScenarioPredictions"
    _field(scenario, "scenario_id", 1, 9)
    _field(scenario, "single_predictions", 2, 11,
           type_name=".waymo.open_dataset.PredictionSet")

    submission = file.message_type.add(); submission.name = "MotionChallengeSubmission"
    _field(submission, "scenario_predictions", 1, 11, label=3,
           type_name=".waymo.open_dataset.ChallengeScenarioPredictions")
    _field(submission, "submission_type", 2, 14,
           type_name=".waymo.open_dataset.MotionChallengeSubmission.SubmissionType")
    enum = submission.enum_type.add(); enum.name = "SubmissionType"
    for name, number in (("UNKNOWN", 0), ("MOTION_PREDICTION", 1),
                         ("INTERACTION_PREDICTION", 2)):
        value = enum.value.add(); value.name, value.number = name, number

    descriptor = descriptor_pool.DescriptorPool().Add(file)
    return (
        message_factory.GetMessageClass(
            descriptor.message_types_by_name["MotionChallengeSubmission"]
        ),
        message_factory.GetMessageClass(
            descriptor.message_types_by_name["ChallengeScenarioPredictions"]
        ),
    )


MotionChallengeSubmission, ChallengeScenarioPredictions = _build_classes()

