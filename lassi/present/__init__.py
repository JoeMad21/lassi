"""Terminal presentation for the `lassi` command line (bible, Readability Standards, Terminal Presentation).

Presentation is optional. The display (the banner) only decorates what a
`lassi` command prints to an interactive terminal and writes no file; the one
file presentation writes is the graphics preset, settings.yaml, and only the
settings step writes it (save_preset, from the first-run prompt or
`lassi settings graphics on|off`). With graphics off every command prints
what it printed without presentation.

- lassi.present.settings: the graphics setting (--graphics, LASSI_GRAPHICS,
  the saved preset, and the first-run prompt) and where the preset lives.
- lassi.present.banner: the banner printed at the start of a command when
  graphics are on.
"""
