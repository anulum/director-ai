// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native Go declaration documentation and directive analysis
// Package main uses the Go parser to bind diagnostics to actual source tokens.
package main

import (
	"crypto/sha256"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

// finding preserves the physical source location rather than adjusted line directives.
type finding struct {
	// Path is the explicit original source input.
	Path string `json:"path"`
	// Rule identifies the parser-based documentation requirement.
	Rule string `json:"rule"`
	// Line is the one-based physical source line.
	Line int `json:"line"`
	// Column is the one-based physical source byte column.
	Column int `json:"column"`
	// Message explains the actual declaration or directive finding.
	Message string `json:"message"`
}

// documented requires actual explanatory text rather than only compiler directives.
func documented(group *ast.CommentGroup) bool {
	if group == nil {
		return false
	}
	explanatory := &ast.CommentGroup{}
	for _, comment := range group.List {
		if !maskingDirective(comment) {
			explanatory.List = append(explanatory.List, comment)
		}
	}
	return strings.TrimSpace(explanatory.Text()) != ""
}

// maskingDirective recognises comment tokens that can alter quality or source positions.
func maskingDirective(comment *ast.Comment) bool {
	text := strings.TrimSpace(strings.TrimPrefix(comment.Text, "//"))
	return strings.HasPrefix(text, "revive:disable") ||
		strings.HasPrefix(comment.Text, "//line ") ||
		strings.HasPrefix(comment.Text, "/*line ")
}

// analyseFile checks all named declarations, including private functions and test entries.
func analyseFile(path string) (string, []finding, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", nil, err
	}
	positions := token.NewFileSet()
	source, err := parser.ParseFile(positions, path, data, parser.ParseComments|parser.AllErrors)
	if err != nil {
		return "", nil, err
	}
	findings := []finding{}
	add := func(position token.Pos, rule, message string) {
		location := positions.PositionFor(position, false)
		findings = append(findings, finding{path, rule, location.Line, location.Column, message})
	}
	for _, group := range source.Comments {
		for _, comment := range group.List {
			if maskingDirective(comment) {
				add(comment.Pos(), "source-directive", "Documentation or source-location masking directive is not allowed")
			}
		}
	}
	for _, declaration := range source.Decls {
		switch node := declaration.(type) {
		case *ast.FuncDecl:
			if !documented(node.Doc) {
				add(node.Name.Pos(), "declaration-documentation", "Function "+node.Name.Name+" requires a documentation comment")
			}
		case *ast.GenDecl:
			if node.Tok == token.IMPORT {
				continue
			}
			for _, specification := range node.Specs {
				switch item := specification.(type) {
				case *ast.TypeSpec:
					if !documented(item.Doc) && !documented(node.Doc) {
						add(item.Name.Pos(), "declaration-documentation", "Type "+item.Name.Name+" requires a documentation comment")
					}
					var fields *ast.FieldList
					switch value := item.Type.(type) {
					case *ast.StructType:
						fields = value.Fields
					case *ast.InterfaceType:
						fields = value.Methods
					}
					if fields != nil {
						for _, field := range fields.List {
							for _, name := range field.Names {
								if ast.IsExported(name.Name) && !documented(field.Doc) && !documented(field.Comment) {
									add(name.Pos(), "member-documentation", "Member "+name.Name+" requires a documentation comment")
								}
							}
						}
					}
				case *ast.ValueSpec:
					if documented(item.Doc) || documented(node.Doc) {
						continue
					}
					for _, name := range item.Names {
						if name.Name != "_" {
							add(name.Pos(), "declaration-documentation", "Value "+name.Name+" requires a documentation comment")
						}
					}
				}
			}
		}
	}
	return fmt.Sprintf("%x", sha256.Sum256(data)), findings, nil
}
