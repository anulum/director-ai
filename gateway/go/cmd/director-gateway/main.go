// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — gateway binary entrypoint

// Command director-gateway runs the Go front door for Director-AI.
// The binary is deliberately small: read config, wire middleware,
// listen. All logic lives in internal/* and is testable without
// spawning a process.
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/anulum/director-ai/gateway/internal/audit"
	"github.com/anulum/director-ai/gateway/internal/config"
	"github.com/anulum/director-ai/gateway/internal/server"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(2)
	}
	var auditLogger *audit.Logger
	if cfg.AuditLogPath != "" {
		auditLogger, err = audit.NewFile(cfg.AuditLogPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "audit log error: %v\n", err)
			os.Exit(2)
		}
		defer auditLogger.Close()
	} else {
		auditLogger = audit.New(os.Stdout)
	}
	log.SetFlags(log.LstdFlags | log.LUTC)
	log.Printf("director-gateway listen=%s upstream=%s api_keys=%d rpm=%d",
		cfg.ListenAddr, cfg.UpstreamURL, len(cfg.APIKeys), cfg.RateLimitRPM)
	if len(cfg.APIKeys) == 0 {
		log.Printf("WARNING: no API keys configured — running in no-auth mode")
	}
	if err := server.Run(cfg, auditLogger); err != nil {
		log.Fatalf("server exited: %v", err)
	}
}
