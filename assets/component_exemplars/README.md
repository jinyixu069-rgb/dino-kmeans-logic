# Component exemplar assets

These images are the generated positive visual prompts used by the current
SAM3 component-search experiment.  They are inputs to the experiment, not
ground-truth masks.

## `screw_bag`

Reference image: `screw_bag/train/good/011.png`.

- `nut.png`: isolate one silver hexagonal nut from the reference, including
  its threaded center; place exactly one centered object on a uniform neutral
  gray studio background; remove every other object, text, and packaging.
- `washer.png`: isolate one flat silver circular washer from the reference;
  place exactly one centered object on a uniform neutral gray studio
  background; remove every other object, text, and packaging.
- `bolt.png`: isolate one complete long silver hex-head threaded bolt from the
  reference; show it horizontally on a uniform neutral gray studio background;
  remove every other object, text, and packaging.

## `pushpins`

Reference image: `pushpins/train/good/037.png`.

- `pushpin.png`: create a clean product photograph containing exactly one
  yellow plastic pushpin matching the reference, including its complete
  spool-shaped plastic body and complete thin silver needle.  Center it on a
  uniform neutral medium-gray background in horizontal side view with generous
  margins.  No box, compartments, other objects, text, logo, or watermark.

All four assets were generated with the built-in image generation tool and
then copied into this directory unchanged.
