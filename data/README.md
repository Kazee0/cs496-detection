# Data

Audio is **not** committed to git. Place WAV files under flat class folders:

```text
data/raw/
  fire_or_smoke_alarm/
  siren/
  glass_breaking/
  scream/
  normal_background/
```

## Google Drive (team clips)

Download or sync `CS396/screams` and `CS396/smoke_alarm` from Google Drive to your machine, then:

```bash
python scripts/import_team_audio.py --source /path/to/CS396
```

Maps:

| Drive folder   | Local class folder              |
|----------------|---------------------------------|
| `smoke_alarm`  | `data/raw/fire_or_smoke_alarm/` |
| `screams`      | `data/raw/scream/`              |

Do **not** upload audio to the GitHub repo — `data/raw/` is gitignored.

## ESC-50 (siren, glass, normal background)

```bash
python scripts/copy_esc50_clips.py glass_breaking siren normal_background
```

Or copy glass/siren manually:

```bash
cp data/ESC-50-master/audio/*-39.wav data/raw/glass_breaking/
cp data/ESC-50-master/audio/*-42.wav data/raw/siren/
python scripts/copy_esc50_clips.py normal_background
```

## Optional room tone

```bash
python scripts/record_background.py
```
