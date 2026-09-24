# Legal and Release Notes

> Engineering/research checklist, not legal advice.

## Separate three questions

1. May the source be accessed/processed for private research?
2. May the source or a derived aligned dataset be redistributed?
3. May weights or a service reproducing an identifiable voice be released?

These are not the same question.

## Public-repository policy

Safe default contents:
- original code
- experiment metadata
- metrics
- benchmark specifications
- links/citations

Do not include without clear permission:
- copyrighted audiobook audio
- copyrighted book text
- DRM-extracted files
- third-party base-model weights
- credentials
- checkpoints trained on rights-unclear voice material

## Voice-specific research

A model intentionally reproducing an identifiable real person's voice raises additional questions beyond dataset copyright. Such checkpoints remain private by default until a release review.

A future explicit licence should consider:
- ML/speech-model training
- underlying text
- sound recording
- performer/voice use
- adapter/model-weight distribution
- synthetic outputs
- commercial/non-commercial use
- territory and duration
- sublicensing/open-source distribution

## Before release

Re-check the law and source terms then in force, including:
- copyright and TDM rules
- database/phonogram/performer rights
- personality/voice rights
- platform subscription terms
- EU AI Act transparency requirements for synthetic audio

Conservative default: publish the **method and evidence** while keeping questionable data and derived checkpoints private; re-run the release candidate on clearly licensed material or obtain permission.
