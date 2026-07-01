# LightBurn File Format Reference

This document is an unofficial working reference for LightBurn project files as
they matter to this repository. It consolidates observations from:

- the local M7 color cards under `LightBurn Colour Card/`
- this repo's importer and writer implementations
- external open-source projects that parse or emit `.lbrn` / `.lbrn2`

It is meant to preserve the current understanding in one place so future work on
`mopa/lightburn_cards.py` and `mopa/lbrn_writer.py` does not depend on memory or
repeating reverse-engineering.

## Scope

- `.lbrn2` is the current XML-based LightBurn project format.
- `.lbrn` is the older XML format. The geometric model is similar, but path data
  is often stored in a more verbose representation.
- This document focuses on the condensed `.lbrn2` structures that this repo
  reads and writes.

## Current repository contract

The repo currently relies on these local modules:

- `mopa/lightburn_cards.py` parses LightBurn color cards and treats their cut
  parameters as canonical machine data.
- `mopa/lbrn_writer.py` writes LightBurn projects and must preserve imported cut
  settings verbatim rather than re-deriving or normalizing them.

That contract matters more than completeness. If LightBurn supports more tags
than listed here, they are out of scope unless this repo needs them.

## Root structure

Observed root tag:

```xml
<LightBurnProject
    AppVersion="1.7.08"
    DeviceName="JCZFiber (LMC4)"
    FormatVersion="1"
    MaterialHeight="0"
    MirrorX="False"
    MirrorY="False">
  ...
</LightBurnProject>
```

Observed root attributes:

- `AppVersion`: LightBurn application version string.
- `DeviceName`: machine label. Present in many real files, not guaranteed.
- `FormatVersion`: currently observed as `1`.
- `MaterialHeight`: machine/material height setting.
- `MirrorX`, `MirrorY`: project-level mirror flags.
- `AskForSendName`: observed in some generated files from external tools.

Common top-level children:

- `CutSetting`
- `CutSetting_Img`
- `Shape`
- `Thumbnail`
- `Notes`
- `VariableText`
- `UIPrefs`

Not every file contains all of them.

## Cut settings

Two closely related layer-setting forms are observed:

### Vector / scan settings

```xml
<CutSetting type="Scan">
  <index Value="0"/>
  <name Value="C00"/>
  <maxPower Value="50"/>
  <speed Value="500"/>
  <frequency Value="300000"/>
  <QPulseWidth Value="20"/>
  <interval Value="0.0012"/>
</CutSetting>
```

### Image settings

```xml
<CutSetting_Img type="Image">
  <index Value="1"/>
  <name Value="Depth"/>
  <numPasses Value="256"/>
  <ditherMode Value="3dslice"/>
  <negative Value="0"/>
  <dpi Value="1270"/>
</CutSetting_Img>
```

Important conventions:

- Child values are usually stored as `<tag Value="..."/>` rather than text.
- `index` is the layer identifier that `Shape/@CutIndex` refers to.
- `type` is commonly `Cut`, `Scan`, or `Image`.
- Real cards often include many more machine parameters than this repo uses.
- For this repo, imported `ColorEntry.raw` values are treated as authoritative.

Fields observed in the supplied M7 color cards or external parsers:

- `index`
- `name`
- `subname`
- `minPower`
- `maxPower`
- `maxPower2`
- `speed`
- `frequency`
- `QPulseWidth`
- `interval`
- `priority`
- `floodFill`
- `bidir`
- `numPasses`
- `dpi`
- `negative`
- `ditherMode`
- `doOutput`
- `overscan`
- `scanOpt`
- `runBlower`
- `hide`

Not every field is present on every layer type.

## Shape model

Observed shape tag:

```xml
<Shape Type="..." CutIndex="...">
  ...
</Shape>
```

Observed shape types across local files and external repositories:

- `Rect`
- `Ellipse`
- `Polygon`
- `Path`
- `Bitmap`
- `Text`
- `Group`

Common shape attributes:

- `Type`
- `CutIndex`
- `VertID`
- `PrimID`
- shape-specific geometry fields like `W`, `H`, `Rx`, `Ry`, `Cr`, `Data`, `Str`

### Rect

```xml
<Shape Type="Rect" CutIndex="0" W="10" H="10" Cr="0">
  <XForm>1 0 0 1 55 55</XForm>
</Shape>
```

- `W`, `H`: width and height.
- `Cr`: corner radius.
- The rectangle is centered on the local origin before transform.

### Ellipse

```xml
<Shape Type="Ellipse" CutIndex="0" Rx="10" Ry="5">
  <XForm>1 0 0 1 55 55</XForm>
</Shape>
```

- `Rx`, `Ry`: radii on each axis.

### Bitmap

```xml
<Shape Type="Bitmap" CutIndex="0" W="39.68" H="39.68" Data="...base64...">
  <XForm>0.252 0 0 0.252 55 55</XForm>
</Shape>
```

Observed bitmap-related attributes:

- `W`, `H`: physical dimensions in mm.
- `Data`: base64-encoded embedded image bytes.
- `File`: external file path or source name in some generators.
- `SourceHash`: hash-like value used by LightBurn.
- `Gamma`, `Contrast`, `Brightness`, `EnhanceAmount`, `EnhanceRadius`,
  `EnhanceDenoise`.

Repository-specific note:

- `mopa/lbrn_writer.py` embeds bitmap bytes into `Data` and also sets the file
  metadata fields so the project remains self-contained.

### Text

Simple form:

```xml
<Shape Type="Text" CutIndex="0" Font="Arial,-1,100,5,50,0,0,0,0,0" Str="Sample" H="25" HasBackupPath="0">
  <XForm>1 0 0 1 0 0</XForm>
</Shape>
```

Observed text-related attributes:

- `Str`
- `Font`
- `H`
- `LS`
- `LnS`
- `Ah`
- `Av`
- `Weld`
- `HasBackupPath`

When `HasBackupPath="1"`, LightBurn may include a nested vector outline:

```xml
<Shape Type="Text" ... HasBackupPath="1">
  <BackupPath Type="Path" CutIndex="0">
    <XForm>1 0 0 1 54.45 56.00</XForm>
    <VertList>...</VertList>
    <PrimList>...</PrimList>
  </BackupPath>
</Shape>
```

Practical implication:

- Text can be treated either as logical text metadata or as geometry via the
  backup path.
- External parsers commonly use `BackupPath` when they need stable geometry.

### Group

```xml
<Shape Type="Group" CutIndex="0">
  <XForm>2 0 0 2 10 10</XForm>
  <Children>
    <Shape Type="Rect" CutIndex="0" W="10" H="10">
      <XForm>1 0 0 1 5 5</XForm>
    </Shape>
  </Children>
</Shape>
```

Observed behavior:

- Groups can nest other groups and shapes.
- Group and child transforms compose.

## XForm

`XForm` is the standard 6-number affine transform payload used by LightBurn:

```xml
<XForm>a b c d e f</XForm>
```

Equivalent matrix form:

```text
[ a  c  e ]
[ b  d  f ]
[ 0  0  1 ]
```

Applied to a point `(x, y)` as:

```text
x' = a*x + c*y + e
y' = b*x + d*y + f
```

Observed conventions:

- Identity is `1 0 0 1 0 0`.
- Pure translation is `1 0 0 1 tx ty`.
- Negative `d` values are often used when mapping image coordinates into
  LightBurn's workspace orientation.
- Group transforms compose with child transforms.

## Path geometry

This is the most important part for `.lbrn2`.

### Legacy `.lbrn` style

Verbose path storage is commonly documented like this:

```xml
<Shape Type="Path" CutIndex="0">
  <XForm>1 0 0 1 0 0</XForm>
  <V vx="0" vy="0"/>
  <V vx="10" vy="0"/>
  <P T="L" p0="0" p1="1"/>
</Shape>
```

That representation is useful conceptually but is not the condensed format this
repo primarily works with.

### Condensed `.lbrn2` style

Observed condensed storage:

```xml
<Shape Type="Path" CutIndex="0" VertID="0" PrimID="0">
  <XForm>1 0 0 1 0 0</XForm>
  <VertList>V0 0V10 0V10 10V0 10</VertList>
  <PrimList>L0 1L1 2L2 3L3 0</PrimList>
</Shape>
```

Observed child elements:

- `VertList`: compact vertex stream
- `PrimList`: compact primitive stream

Observed shape attributes:

- `VertID`
- `PrimID`

These appear to support geometry reuse or caching across shapes. External
parsers commonly resolve them by caching previously seen `VertList` / `PrimList`
payloads.

### VertList

Observed compact grammar:

- Vertices start with `V`
- A vertex contains `x y`
- Optional Bezier control-point fragments may follow on the same token stream:
  - `c0x`
  - `c0y`
  - `c1x`
  - `c1y`

Example:

```text
V49 48c0x1c1x49c1y48V62 63c0x62c0y63c1x1
```

Interpretation used by external tooling:

- vertex position is given by the `Vx y` pair
- `c0*` describes the outgoing Bezier control point from that vertex
- `c1*` describes the incoming Bezier control point to that vertex

Observed numeric patterns:

- integers
- decimals
- negative values
- scientific notation

### PrimList

Observed primitive forms:

- `Lstart end`: line segment
- `Bstart end`: cubic Bezier segment
- `LineClosed`: special closed-path marker seen in some text/path fixtures

Examples:

```text
L0 1
B0 1
L0 1B1 2L2 3B3 0
LineClosed
```

Practical notes:

- indices refer into the parsed `VertList` vertex array
- mixed `L` and `B` primitives are normal
- some generators close a path by emitting the final `Llast 0`
- some files use `LineClosed` instead

## Thumbnail and Notes

### Thumbnail

Most commonly observed form:

```xml
<Thumbnail Source="...base64..."/>
```

Alternative forms are handled by some external extractors, but the attribute
form is the dominant one observed in real files.

### Notes

Observed forms:

```xml
<Notes ShowOnLoad="0" Notes="line 1&#10;line 2"/>
```

and

```xml
<Notes>freeform text</Notes>
```

Both forms should be treated as valid when extracting text.

## Minimal synthesized example

This is not copied from a LightBurn export. It is a synthesized example based on
 observed conventions and is suitable as a mental model:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<LightBurnProject AppVersion="1.7.08" FormatVersion="1" MaterialHeight="0" MirrorX="False" MirrorY="False">
  <Thumbnail Source="...base64 png..."/>

  <CutSetting type="Scan">
    <index Value="0"/>
    <name Value="C00"/>
    <maxPower Value="50"/>
    <speed Value="500"/>
    <frequency Value="300000"/>
    <QPulseWidth Value="20"/>
    <interval Value="0.0012"/>
  </CutSetting>

  <CutSetting_Img type="Image">
    <index Value="1"/>
    <name Value="Depth"/>
    <numPasses Value="256"/>
    <ditherMode Value="3dslice"/>
    <negative Value="0"/>
    <dpi Value="1270"/>
  </CutSetting_Img>

  <Shape Type="Path" CutIndex="0" VertID="0" PrimID="0">
    <XForm>1 0 0 1 0 0</XForm>
    <VertList>V0 0V10 0V10 10V0 10</VertList>
    <PrimList>L0 1L1 2L2 3L3 0</PrimList>
  </Shape>

  <Shape Type="Bitmap" CutIndex="1" W="25" H="25" Gamma="1" Contrast="0" Brightness="0" File="depth.png" SourceHash="123" Data="...base64 png...">
    <XForm>0.1 0 0 -0.1 0 25</XForm>
  </Shape>

  <Notes ShowOnLoad="0" Notes="Generated by mopa-heightmap"/>
</LightBurnProject>
```

## What appears stable enough to rely on

The following appear consistently enough across local files and external repos to
be treated as stable assumptions for this codebase:

- root tag `LightBurnProject`
- project version `FormatVersion="1"`
- layer blocks stored as `CutSetting` or `CutSetting_Img`
- child setting values stored on `Value` attributes
- shape routing through `Shape/@Type` and `Shape/@CutIndex`
- affine transforms stored in `XForm`
- condensed path storage through `VertList` and `PrimList`
- bitmap embedding through `Shape Type="Bitmap"` with base64 `Data`
- thumbnail storage through `<Thumbnail Source="...">`
- text storage through `Shape Type="Text"` and `Str`

## What is still uncertain

The following are known gaps:

- complete meaning of every cut-setting child LightBurn may emit
- full semantics of `CutSetting_Img` variants beyond the image-mode fields we use
- exact reuse rules around `VertID` / `PrimID`
- less common shape types not seen in the supplied cards or surveyed repos
- whether all LightBurn versions accept exactly the same image defaults and
  auxiliary tags

If future work depends on those areas, update this document and add fixtures or
tests in the same change.

## External repositories consulted

These were particularly useful when consolidating the format:

- `jlucaso1/lbrn2-to-svg`: practical parser/types for condensed `.lbrn2`
- `styx/lac_to_lbrn2`: direct `.lbrn2` emitter covering `Path`, `Ellipse`, and
  `Bitmap`
- `RichGibson/maker-file-index`: metadata, thumbnail, notes, and path parsing
- `ranaur/lightburn-cli`: conceptual format notes, especially around shapes and
  transforms
- `LordBex/LightBurn-Browser`: confirmation of real-world thumbnail/font
  extraction patterns