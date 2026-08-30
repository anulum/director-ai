SPDX-License-Identifier: Apache-2.0
Commercial license available
Copyright 2020-2026 Miroslav Sotek
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI - Offline Licence Key Ceremony Runbook

Director-AI SEC-1 offline key ceremony
=======================================

This bundle generates the production Ed25519 licence-signing keypair.

Before running
--------------
1. Verify the laptop was freshly booted from trusted media and has no network.
2. Disable Wi-Fi and Bluetooth and unplug Ethernet.
3. Insert the ceremony bundle medium.
4. Insert a separate PRIVATE vault medium and a PUBLIC transfer medium.
5. Windows: double-click RUN_KEY_CEREMONY_WINDOWS.cmd.
   Linux: run or double-click RUN_KEY_CEREMONY_LINUX.sh.

Outputs
-------
- PRIVATE vault: director_license_private_key.hex
- PUBLIC transfer: PUBLIC_KEY_ONLY.txt

The private file must never be connected to an online system, committed to
Git, pasted into chat, placed in logs, or copied to the public transfer medium.
The controller rejects output folders reported by the operating system as the
same filesystem device; do not use two partitions on one physical device.
Only PUBLIC_KEY_ONLY.txt returns to the canonical workstation. Keep the private
medium physically controlled and make an encrypted offline backup under a
separate custody path.

The manifest detects accidental corruption. It is not a digital signature and
does not replace custody of the bundle between preparation and execution.
