// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native TypeScript source documentation command

import { createHash } from "node:crypto";
import { readFileSync, realpathSync, statSync } from "node:fs";
import { resolve, relative as relativePath, sep } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

/** Actual parser configuration; semantic/import qualification is a separate profile. */
const compilerOptions = {
  allowJs: true,
  noEmit: true,
  noLib: true,
  noResolve: true,
  jsx: ts.JsxEmit.Preserve,
  target: ts.ScriptTarget.Latest,
};

/** Named declaration forms whose documentation participates in the source ratchet. */
const declarations = new Set([
  ts.SyntaxKind.FunctionDeclaration, ts.SyntaxKind.MethodDeclaration,
  ts.SyntaxKind.MethodSignature, ts.SyntaxKind.Constructor,
  ts.SyntaxKind.GetAccessor, ts.SyntaxKind.SetAccessor,
  ts.SyntaxKind.ClassDeclaration, ts.SyntaxKind.InterfaceDeclaration,
  ts.SyntaxKind.TypeAliasDeclaration, ts.SyntaxKind.EnumDeclaration,
  ts.SyntaxKind.EnumMember, ts.SyntaxKind.ModuleDeclaration,
  ts.SyntaxKind.PropertyDeclaration, ts.SyntaxKind.PropertySignature,
  ts.SyntaxKind.CallSignature, ts.SyntaxKind.ConstructSignature,
]);

/**
 * Identify actual named variables without demanding comments on local data bindings.
 * @param {ts.Node} node Actual native source node.
 * @returns {boolean} Whether its declaration requires explanatory documentation.
 */
function requiresDocumentation(node) {
  if (declarations.has(node.kind)) return true;
  return ts.isVariableDeclaration(node) &&
    ((node.initializer !== undefined &&
      (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))) ||
      (ts.isVariableStatement(node.parent.parent) &&
        node.parent.parent.parent.kind === ts.SyntaxKind.SourceFile));
}

/**
 * Require explanatory JSDoc text rather than a source masking or type-only annotation.
 * @param {ts.Node} node Actual native declaration.
 * @returns {boolean} Whether native JSDoc retains explanatory text.
 */
function documented(node) {
  const comments = [...ts.getJSDocCommentsAndTags(node)];
  if (ts.isVariableDeclaration(node) && ts.isVariableStatement(node.parent.parent)) {
    comments.push(...ts.getJSDocCommentsAndTags(node.parent.parent));
  }
  return comments.flatMap((comment) =>
    ts.isJSDoc(comment) && comment.tags !== undefined
      ? [comment, ...comment.tags] : [comment]).some((comment) =>
    (typeof comment.comment === "string"
      ? comment.comment.trim().length > 0
      : comment.comment !== undefined && comment.comment.length > 0));
}

/**
 * Analyse complete original source inputs through the actual native compiler parser.
 * @param {string} root Canonical selected source root.
 * @param {string[]} paths Explicit canonical maintained source inputs.
 * @returns {{schema: string, complete: boolean, sources: Record<string,string>,
 * findings: {path:string, rule:string, line:number, column:number, message:string}[]}}
 * Complete source hashes and original parser-derived documentation findings.
 */
export function captureDocumentation(root, paths) {
  root = realpathSync(root);
  if (paths.length === 0 || new Set(paths).size !== paths.length) {
    throw new Error("Explicit nonempty unique source inputs are required");
  }
  /**
   * Original source inputs bound to their exact native byte digests.
   * @type {Record<string, string>}
   */
  const sources = {};
  for (const path of paths) {
    const actual = resolve(root, path);
    if (relativePath(root, actual).split(sep).join("/") !== path ||
      realpathSync(actual) !== actual || !statSync(actual).isFile()) {
      throw new Error("Source input is not canonical and regular");
    }
    sources[path] = createHash("sha256").update(readFileSync(actual)).digest("hex");
  }
  const program = ts.createProgram(paths.map((path) => resolve(root, path)), compilerOptions);
  program.getTypeChecker();
  /** Complete source-linked original native findings; no count-only projection. */
  const findings = [];
  const syntax = program.getSyntacticDiagnostics();
  if (syntax.length > 0) {
    for (const diagnostic of syntax) {
      const position = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
      findings.push({
        path: relativePath(root, diagnostic.file.fileName).split(sep).join("/"),
        rule: "TS" + diagnostic.code, line: position.line + 1,
        column: position.character + 1,
        message: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
      });
    }
    return { schema: "quality-ratchet.ts-ast.v1", complete: false, sources, findings };
  }
  for (const path of paths) {
    const parsed = program.getSourceFile(resolve(root, path));
    if (parsed === undefined) throw new Error("Native parser omitted a source input");
    const source = parsed;
    /**
     * Actual comment token offsets, deduplicated across their native AST owners.
     * @type {Map<number, ts.CommentRange>}
     */
    const comments = new Map();
    /**
     * Original JSX text spans whose comment-looking content remains literal text.
     * @type {{pos: number, end: number}[]}
     */
    const jsxText = [];
    /**
     * Visit native source nodes while retaining their original line and column.
     * @param {ts.Node} node Original parsed source node.
     * @returns {void} Findings are appended to the complete source carrier.
     */
    function visit(node) {
      if (requiresDocumentation(node) && !documented(node)) {
        const position = source.getLineAndCharacterOfPosition(node.getStart(source));
        findings.push({
          path, rule: "declaration-documentation", line: position.line + 1,
          column: position.character + 1,
          message: ts.SyntaxKind[node.kind] + " requires explanatory JSDoc",
        });
      }
      if (ts.isJsxText(node)) {
        jsxText.push({ pos: node.pos, end: node.end });
      } else {
        for (const range of [
          ...(ts.getLeadingCommentRanges(source.text, node.pos) ?? []),
          ...(ts.getTrailingCommentRanges(source.text, node.end) ?? []),
        ]) comments.set(range.pos, range);
      }
      ts.forEachChild(node, visit);
    }
    visit(source);
    for (const comment of comments.values()) {
      if (jsxText.some(({ pos, end }) => comment.pos >= pos && comment.pos < end)) continue;
      const text = source.text.slice(comment.pos, comment.end);
      if (/^\s*(?:\/\/|\/\*+|\*)?\s*@(?:ts-ignore|ts-nocheck|ts-expect-error|hidden|ignore)\b/m.test(text)) {
        const position = source.getLineAndCharacterOfPosition(comment.pos);
        findings.push({
          path, rule: "source-directive", line: position.line + 1,
          column: position.character + 1,
          message: "Documentation or typing masking directive is not allowed",
        });
      }
    }
  }
  for (const path of paths) {
    const current = createHash("sha256").update(readFileSync(resolve(root, path))).digest("hex");
    if (current !== sources[path]) throw new Error("Native source changed during analysis");
  }
  return { schema: "quality-ratchet.ts-ast.v1", complete: true, sources, findings };
}

/**
 * Run the finite native command without substituting a successful report on failure.
 * @param {string[]} args Original process arguments.
 * @returns {number} Zero for clean, one for findings and two for incomplete analysis.
 */
export function main(args) {
  if (args.length === 1 && args[0] === "--version") {
    console.log(JSON.stringify({ node: process.versions.node, typescript: ts.version,
      profile: "typescript-docs", compilerOptions }));
    return 0;
  }
  try {
    const [root, ...paths] = args;
    if (root === undefined) throw new Error("Source root is required");
    const result = captureDocumentation(root, paths);
    console.log(JSON.stringify(result));
    return result.complete ? Number(result.findings.length > 0) : 2;
  } catch {
    console.error("Native TypeScript documentation refused.");
    return 2;
  }
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
