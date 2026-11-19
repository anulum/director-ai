-- SPDX-License-Identifier: Apache-2.0
-- Commercial license available
-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
-- © Code 2020–2026 Miroslav Šotek. All rights reserved.
-- ORCID: 0009-0009-3560-0851
-- Contact: www.anulum.li | protoscience@anulum.li
-- Director-AI — native Lean source syntax and documentation reader
import Lean

/-!
Native source inspection for the quality ratchet. The reader preserves the
compiler's source syntax, documentation, coordinates and diagnostics; source
ownership and historical allowances are checked by the calling profile.
-/

open Lean

/-- Preserve the compiler's original coordinates and UTF-8 byte offset. -/
def positionJson (fileMap : FileMap) (p : String.Pos.Raw) : Json :=
  let position := fileMap.toPosition p
  Json.mkObj [
    ("line", toJson position.line), ("column", toJson position.column),
    ("byte_offset", toJson p.byteIdx)]

/-- Preserve both native declaration and identifier-selection ranges. -/
def rangeJson (ranges : DeclarationRanges) : Json :=
  Json.mkObj [
    ("line", toJson ranges.range.pos.line),
    ("column", toJson ranges.range.pos.column),
    ("end_line", toJson ranges.range.endPos.line),
    ("end_column", toJson ranges.range.endPos.column),
    ("selection_line", toJson ranges.selectionRange.pos.line),
    ("selection_column", toJson ranges.selectionRange.pos.column),
    ("selection_end_line", toJson ranges.selectionRange.endPos.line),
    ("selection_end_column", toJson ranges.selectionRange.endPos.column)]

/-- Retain actual syntax kinds, lexical leaves, children and nullable native positions. -/
partial def syntaxJson (fileMap : FileMap) (stx : Syntax) : Json :=
  Json.mkObj [
    ("kind", toJson stx.getKind.toString),
    ("identifier", if stx.isIdent then toJson stx.getId.toString else Json.null),
    ("atom", if stx.isAtom then toJson stx.getAtomVal else Json.null),
    ("start", stx.getPos?.map (positionJson fileMap) |>.getD Json.null),
    ("end", stx.getTailPos?.map (positionJson fileMap) |>.getD Json.null),
    ("children", Json.arr (stx.getArgs.map (syntaxJson fileMap)))]

/-- Preserve all reported and unreported native messages, including original text. -/
def messagesJson (log : MessageLog) : IO Json := do
  let rows ← log.reportedPlusUnreported.toArray.mapM fun message => do
    return Json.mkObj [
      ("severity", toJson message.severity),
      ("line", toJson message.pos.line), ("column", toJson message.pos.column),
      ("message", toJson (← message.data.toString))]
  return Json.arr rows

/-- Read original source headers or the complete elaborated native command environment. -/
unsafe def readSource (mode fileName sysroot : String) (moduleName : Name) : IO UInt32 := do
  initSearchPath sysroot
  let source ← IO.FS.readFile fileName
  let inputCtx := Parser.mkInputContext source fileName
  let (header, parserState, headerMessages) ← Parser.parseHeader inputCtx
  let imports :=
    if headerMessages.hasErrors then #[]
    else (Elab.HeaderSyntax.imports header).map (·.module.toString)
  let mut commands : Array Syntax := #[]
  let mut constants : Array Json := #[]
  let mut messages := headerMessages
  if mode == "inspect" && !headerMessages.hasErrors then
    enableInitializersExecution
    let (env, importedMessages) ← Elab.processHeader header {} headerMessages inputCtx
      (mainModule := moduleName)
    let state ← Elab.IO.processCommands inputCtx parserState
      (Elab.Command.mkState env importedMessages {})
    commands := state.commands
    messages := state.commandState.messages
    let env := state.commandState.env
    for (name, _) in env.constants.toList do
      if (env.getModuleIdxFor? name).isNone then
        -- Local declarations use the same extension state at every import level.
        let ranges := declRangeExt.find? env name
        let doc ← findDocString? env name (includeBuiltin := false)
        constants := constants.push (Json.mkObj [
          ("name", toJson name.toString), ("doc", toJson doc),
          ("range", ranges.map rangeJson |>.getD Json.null)])
  IO.println (Json.mkObj [
    ("schema", toJson ("quality-ratchet.lean-native.v1" : String)),
    ("compiler", toJson Lean.versionString), ("compiler_commit", toJson Lean.githash),
    ("mode", toJson mode), ("module", toJson moduleName.toString),
    ("source", toJson source), ("has_errors", toJson messages.hasErrors),
    ("header", syntaxJson inputCtx.fileMap header.raw),
    ("imports", toJson imports),
    ("messages", ← messagesJson messages),
    ("commands", Json.arr (commands.map (syntaxJson inputCtx.fileMap))),
    ("constants", Json.arr constants)] |>.compress)
  return if messages.hasErrors then 2 else 0

/-- Require an explicit mode, original source path, module identity and toolchain root. -/
unsafe def main (args : List String) : IO UInt32 := do
  match args with
  | [mode, fileName, moduleName, sysroot] =>
    if (mode == "header" || mode == "inspect") && !moduleName.isEmpty && !sysroot.isEmpty then
      readSource mode fileName sysroot moduleName.toName
    else
      IO.eprintln "Expected header or inspect and nonempty module and toolchain identities"
      return 2
  | _ =>
    IO.eprintln "Expected mode, source path, module identity and toolchain root"
    return 2
