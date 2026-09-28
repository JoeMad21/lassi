"""Terminal presentation for the `lassi` command line (bible, Readability Standards, Terminal Presentation).

Presentation is optional. The display (the banner and the live inference
table) only decorates what a `lassi` command prints and writes no file; the
one file presentation writes is the graphics preset, settings.yaml, and
only the settings step writes it (save_preset, from the prompt, until a
preset is saved, or `lassi settings graphics on|off`). With graphics off
every command prints what it printed without presentation.

- lassi.present.settings: the graphics setting (--graphics, LASSI_GRAPHICS,
  the saved preset, and the prompt, until a preset is saved) and where the
  preset lives.
- lassi.present.banner: the banner printed at the start of a command when
  graphics are on.
- lassi.present.live: the live inference table `lassi run` draws when
  graphics are on, as its progress observer (lassi.core.progress).
"""
