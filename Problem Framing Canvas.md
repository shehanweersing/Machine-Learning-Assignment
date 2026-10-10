# Problem framing canvas

| Item | Project decision |
|---|---|
| Stakeholder | Mathematics teachers and school counsellors |
| Decision need | Decide which students should receive a contextual academic-support review after the second grading period |
| Primary lens | Regression |
| Unit of analysis | One student in the mathematics course |
| Input availability | Demographic, family, social, lifestyle, absence, failure, `G1` and `G2` information available after the second period |
| Output | Predicted final mathematics grade `G3` on a 0-20 scale |
| Success criteria | Low RMSE and MAE on an untouched test set, with cross-validation used for model selection |
| Operational use | Use predictions as a screening signal for human review, not as an automatic grade or disciplinary decision |
| Primary trade-off | `G1` and `G2` improve accuracy but mean this is not a pre-assessment intervention model |

## Scope boundary

The main model answers: “Given information available after the second
period, what final mathematics grade is plausible for this student?” It does
not answer whether a feature causes a grade, whether an intervention will
work, or whether a student should be punished or excluded.

An early-intervention sensitivity run excludes `G1`, `G2`, and
`grade_progress`. Its results are reported separately because its prediction
timing and performance are materially different.
