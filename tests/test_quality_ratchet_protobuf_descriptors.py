# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Protobuf declaration and comment integration tests
"""Exercise original descriptor structure through actual compiler source analysis."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_protobuf import capture_protobuf_documentation


def test_native_group_descriptor_transport_loss_refuses(
    tmp_path: Path,
) -> None:
    """Retain actual compiled group bytes before deliberate type-link corruption.

    Parameters
    ----------
    tmp_path : Path
        Original native schema, wrapper and process-output custody.
    """
    root = tmp_path / "group-source"
    (root / "proto").mkdir(parents=True)
    (root / "proto/group.proto").write_text(
        '// Group wire.\nsyntax = "proto2";\npackage admission;\n'
        "// Source payload.\nmessage Request {\n"
        "  // Original group.\n  optional group Payload = 1 {\n"
        "    // Original text.\n    optional string text = 2;\n  }\n}\n"
    )
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    assert not capture_protobuf_documentation(root, "protoc").findings
    protoc = shutil.which("protoc")
    assert protoc is not None
    wrapper = tmp_path / "protoc-original-before-group-loss"
    wrapper.write_text(
        "#!" + sys.executable + "\n"
        "import sys,subprocess,json,base64\nfrom pathlib import Path\n"
        "from google.protobuf import descriptor_pb2\n"
        "args=sys.argv[1:]; result=subprocess.run(["
        + repr(protoc)
        + ",*args],capture_output=True,timeout=20)\n"
        "with Path('native-group-original.jsonl').open('a') as stream:\n"
        " stream.write(json.dumps({'status':result.returncode,"
        "'stdout_b64':base64.b64encode(result.stdout).decode()})+'\\n')\n"
        "output=result.stdout\n"
        "if args!=['--version']:\n"
        " descriptor=descriptor_pb2.FileDescriptorSet.FromString(output)\n"
        " descriptor.file[0].message_type[0].field[0].type_name='.admission.Absent'\n"
        " output=descriptor.SerializeToString()\n"
        "sys.stdout.buffer.write(output);sys.stderr.buffer.write(result.stderr)\n"
        "raise SystemExit(result.returncode)\n"
    )
    wrapper.chmod(0o700)
    with pytest.raises(ValueError, match="group field has no original type"):
        capture_protobuf_documentation(root, str(wrapper))
    originals = [
        json.loads(line)
        for line in (root / "native-group-original.jsonl").read_text().splitlines()
    ]
    assert len(originals) == 2 and all(item["status"] == 0 for item in originals)
    assert all(base64.b64decode(item["stdout_b64"]) for item in originals)


DOCUMENTED_SCHEMA = """// Admission wire contract.
syntax = "proto3";
package admission;
// Actual admission request.
message Request {
  // Original caller text.
  string text = 1;
}
"""


@pytest.fixture
def protobuf_repository(tmp_path: Path) -> Path:
    """Create a real committed native schema source repository.

    Parameters
    ----------
    tmp_path : Path
        Actual isolated Samsung runner directory.

    Returns
    -------
    Path
        Committed original maintained source root.
    """
    root = tmp_path / "protobuf-source"
    (root / "proto").mkdir(parents=True)
    (root / "proto/admission.proto").write_text(DOCUMENTED_SCHEMA)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Native fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "original schema",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def test_native_protobuf_all_declaration_forms_and_synthetic_boundaries(
    protobuf_repository: Path,
) -> None:
    """Measure nested APIs, real oneofs, maps, optional fields and streaming RPCs.

    Parameters
    ----------
    protobuf_repository : Path
        Actual native schema repository.
    """
    path = protobuf_repository / "proto/admission.proto"
    path.write_text("""// Complete wire API.
syntax = "proto3";
package admission;
// Wrapper request.
message Request {
  // Map of caller attributes.
  map<string, string> attributes = 1;
  // Optional caller identity.
  optional string identity = 2;
  // Selection of source content.
  oneof content {
    // Native text content.
    string text = 3;
  }
  // Nested event.
  message Event {
    // Actual event id.
    string id = 1;
  }
  // Nested state.
  enum State {
    // Unknown state.
    UNKNOWN = 0;
  }
}
// Top-level status.
enum Status {
  // Initial status.
  INITIAL = 0;
}
// Admission transport.
service Admission {
  // Review actual requests.
  rpc Review(stream Request) returns (stream Request);
}
""")
    clean = capture_protobuf_documentation(protobuf_repository, "protoc")
    assert not clean.findings
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not line.lstrip().startswith("//")
        )
        + "\n"
    )
    actual = capture_protobuf_documentation(protobuf_repository, "protoc")
    names = {
        finding.message.removeprefix("Missing attached documentation for ")
        for finding in actual.findings
    }
    assert names == {
        "schema file",
        "message Request",
        "field attributes",
        "field identity",
        "oneof content",
        "field text",
        "message Event",
        "field id",
        "enum State",
        "enum value UNKNOWN",
        "enum Status",
        "enum value INITIAL",
        "service Admission",
        "RPC Review",
    }


def test_native_proto2_extensions_groups_and_multiple_extend_locations(
    protobuf_repository: Path,
) -> None:
    """Preserve genuine proto2 extension and group source locations.

    Parameters
    ----------
    protobuf_repository : Path
        Actual native source root.
    """
    path = protobuf_repository / "proto/admission.proto"
    path.write_text("""// Extension wire contract.
syntax = "proto2";
package admission;
// Extensible input.
message Request {
  extensions 100 to 200;
  extend Request {
    // Nested extension input.
    optional bool nested = 102;
  }
  // Legacy grouped payload.
  optional group Payload = 1 {
    // Original text.
    optional string text = 2;
  }
}
extend Request {
  // First additional input.
  optional string first = 100;
}
extend Request {
  // Second additional input.
  optional string second = 101;
  // Additional grouped payload.
  optional group Extra = 103 {
    // Grouped extension value.
    optional string value = 1;
  }
}
""")
    actual = capture_protobuf_documentation(protobuf_repository, "protoc")
    assert not actual.findings
    path.write_text(path.read_text().replace("// Second additional input.", ""))
    actual = capture_protobuf_documentation(protobuf_repository, "protoc")
    assert any(
        finding.message.endswith("extension second") for finding in actual.findings
    )


def test_native_protobuf_imports_and_comment_attachment(
    protobuf_repository: Path,
) -> None:
    """Keep every real import while refusing detached-only documentation credit.

    Parameters
    ----------
    protobuf_repository : Path
        Actual maintained source repository.
    """
    path = protobuf_repository / "proto/admission.proto"
    path.write_text(
        DOCUMENTED_SCHEMA.replace(
            "// Original caller text.\n", "// Detached input description.\n\n"
        )
    )
    other = protobuf_repository / "proto/response.proto"
    other.write_text("""// Review response wire.
syntax = "proto3";
package admission;
import "proto/admission.proto";
// Returned source.
message Response {
  Request original = 1; // Original submitted request.
}
""")
    actual = capture_protobuf_documentation(protobuf_repository, "protoc")
    assert len(actual.sources) == 2 and len(actual.findings) == 1
    assert actual.findings[0].message.endswith("field text")


@pytest.mark.parametrize(
    "comment", ("@exclude hidden input", "buf:lint:ignore COMMENT_FIELD")
)
def test_native_protobuf_real_comment_masks_cannot_hide_debt(
    protobuf_repository: Path,
    comment: str,
) -> None:
    """Recognize actual attached comment directives without parsing source strings.

    Parameters
    ----------
    protobuf_repository : Path
        Actual schema source root.
    comment : str
        Documentation mask attached to a real declaration.
    """
    path = protobuf_repository / "proto/admission.proto"
    path.write_text(DOCUMENTED_SCHEMA.replace("Original caller text.", comment))
    actual = capture_protobuf_documentation(protobuf_repository, "protoc")
    assert {finding.rule for finding in actual.findings} == {
        "source-directive",
        "declaration-documentation",
    }
    path.write_text(
        DOCUMENTED_SCHEMA.replace(
            "package admission;",
            'package admission;\noption java_package = "@exclude.example";',
        )
    )
    assert not capture_protobuf_documentation(protobuf_repository, "protoc").findings
