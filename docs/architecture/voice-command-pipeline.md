# Voice command pipeline prototype

## Data path

```text
microphone or PCM WAV
        ↓
optional Vosk constrained-grammar recognizer
        ↓ transcript + word-confidence summary
allowlisted phrase parser
        ↓ typed SkillIntent
confidence + skill-availability gate
        ↓ SkillDispatch (e.g. request balance / safe-stop event)
future behavior supervisor
```

The recognizer is an optional adapter in `src/hitmonlee_sim/voice.py`. The parser and dispatcher in `commands.py` do not depend on Vosk, so a different local or streaming ASR engine can replace it without changing the skill schema.

## Current implementation

- The `speech` optional extra installs Vosk and `sounddevice`.
- The default Vosk model is `vosk-model-small-en-us-0.15`. Download it with `python -m hitmonlee_sim.voice --download-model`; the approximately 40 MB model and a `SOURCE.txt` license/provenance note live under ignored `runs/asr/`.
- WAV input must be mono, signed 16-bit PCM at 16 kHz. The microphone path requests the same format from PortAudio.
- Vosk is initialized with a short list of robot command phrases, common negated variants and its unknown token. The final recognizer result is normalized to `SkillIntent`; any detected negation causes rejection rather than stripping it and accepting the remaining command.
- `STAND` requests the registered `balance` skill. `STOP` creates a supervisor safe-stop request. Walking and turns are rejected unless a corresponding skill is registered. Unknown, negated and low-confidence transcripts fail closed.
- The current confidence is the mean of Vosk's word confidence values. Treat it as a heuristic; it has not been calibrated as a probability or evaluated on a real-speaker dataset.

## Verified example

A Windows speech synthesizer produced 16 kHz mono test WAVs saying “stand”, “walk forward”, “stop”, and “don't stop”. Vosk transcribed them at confidence 1.0; the gate respectively returned `accepted / balance`, `skill_unavailable`, `accepted / supervisor.safe_stop_request`, and `unknown_intent`. This checks file decoding and routing; it does not measure recognition accuracy for a human voice, room noise, far-field microphones, or adversarial audio. A microphone input device is present, but user-specific capture quality still needs a live trial.

## Learning and safety boundary

Run `python -m hitmonlee_sim.voice --listen` after model setup to capture one local utterance. Inspect transcript, confidence and dispatch status before changing the grammar. Do not lower the gate simply to make more examples pass. Record a labelled set of positive commands, negations, ambiguous phrases, different speakers and room conditions; select thresholds from false accepts and false rejects.

The dispatch output is a skill request, not a motor command. The balance policy does not yet subscribe to this request. Any physical robot must have an independent emergency-stop circuit and a deterministic supervisor that validates state, timing, limits and watchdog behavior.

Upstream references: [Vosk API](https://github.com/alphacep/vosk-api), [audio-format requirements](https://alphacephei.com/vosk/install), [model catalog and license information](https://alphacephei.com/vosk/models.html).
