#!/usr/bin/env python3
"""Compiler for the simple.vm assembly language.

The emitted bytecode intentionally stays compatible with the original Perl
compiler.  Metadata is kept outside the bytecode, so it is safe for existing
VMs and tools to consume compiled files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


OPCODES = {
    "exit": 0x00, "int_store": 0x01, "int_print": 0x02,
    "int_tostring": 0x03, "int_random": 0x04,

    "jmp": 0x10, "jmpz": 0x11, "jmpnz": 0x12,

    "xor": 0x20, "add": 0x21, "sub": 0x22, "mul": 0x23,
    "div": 0x24, "inc": 0x25, "dec": 0x26, "and": 0x27, "or": 0x28,

    "string_store": 0x30, "string_print": 0x31, "concat": 0x32,
    "system": 0x33, "string_toint": 0x34,

    "cmp_reg": 0x40, "cmp_immediate": 0x41, "cmp_string": 0x42,
    "is_string": 0x43, "is_integer": 0x44,

    "nop": 0x50, "reg_store": 0x51,

    "peek": 0x60, "poke": 0x61, "memcpy": 0x62,

    "push": 0x70, "pop": 0x71, "ret": 0x72, "call": 0x73,
}

ADDRESS_RE = re.compile(r"0[xX][0-9a-fA-F]+|[0-9]+$")


class CompileError(Exception):
    """Raised for an invalid source program or output operation."""


def parse_number(text: str) -> int:
    return int(text, 16) if text.lower().startswith("0x") else int(text, 10)


def is_address(text: str) -> bool:
    return ADDRESS_RE.fullmatch(text) is not None


def encode_string(text: str) -> bytes:
    """Preserve the two escapes accepted by the legacy compiler."""
    try:
        return text.replace(r"\n", "\n").replace(r"\t", "\t").encode("latin-1")

    except UnicodeEncodeError as exc:
        raise CompileError(
            "string contains characters outside the byte range",
        ) from exc


@dataclass
class Label:
    name: str
    line: int
    aliases: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "source": {"line": self.line},
        }


@dataclass
class PendingAddress:
    offset: int
    label: str


class BytecodeWriter:
    """Owns bytecode construction and deferred symbolic-address patches."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.labels: dict[int, Label] = {}
        self.pending: list[PendingAddress] = []

    @property
    def offset(self) -> int:
        return len(self.data)

    def emit(self, *values: int) -> None:
        for value in values:
            if not 0 <= value <= 0xFF:
                raise CompileError(f"byte must fit in one byte: {value}")

            self.data.append(value)

    def emit_word(self, value: int, description: str) -> None:
        if not 0 <= value <= 0xFFFF:
            raise CompileError(f"{description} must fit in 16 bits: {value}")

        self.data.extend(value.to_bytes(2, "little"))

    def emit_target(self, opcode: int, target: str) -> None:
        self.emit(opcode)

        if is_address(target):
            self.emit_word(parse_number(target), "address")
        else:
            self.pending.append(PendingAddress(self.offset, target))
            self.data.extend(b"\0\0")

    def define_label(self, name: str, line: int) -> None:
        name_already_taken = False
        for label in self.labels.values():
            if name == label.name or name in label.aliases:
                name_already_taken = True
                break

        if name_already_taken:
            print(
                f"WARNING: Label name '{name}' defined multiple times!",
                file=sys.stderr,
            )
            print("         Picking first occurrence.", file=sys.stderr)

            return

        if re.match(r"^(?:0x|[0-9]+$)", name, re.IGNORECASE):
            print(
                f"WARNING: Label named '{name}' WILL be confused for an address",
                file=sys.stderr,
            )

        if self.offset in self.labels:
            self.labels[self.offset].aliases.append(name)
        else:
            self.labels[self.offset] = Label(name, line)

    def finish(self) -> bytes:
        names = {label.name: offset for offset, label in self.labels.items()}

        for offset, label in self.labels.items():
            for alias in label.aliases:
                names[alias] = offset
        
        for patch in self.pending:
            if patch.label not in names:
                raise CompileError(
                    f"No target for label '{patch.label}' - Label not defined!",
                )

            self.data[patch.offset:patch.offset + 2] = \
                names[patch.label].to_bytes(2, "little")

        return bytes(self.data)

    def label_metadata(self) -> dict[str, dict[str, object]]:
        res = {}
        for offset, label in self.labels.items():
            res[str(offset)] = label.metadata()
        
        return res


class AssemblyCompiler:
    """Turns one assembly source file into a ``BytecodeWriter`` result."""

    THREE_REGISTER_OPS = {
        name: OPCODES[name] for name in 
            ("add", "and", "sub", "mul", "div", "or", "xor", "concat")
    }

    ONE_REGISTER_OPS = {
        "dec": OPCODES["dec"], "inc": OPCODES["inc"],
        "int2string": OPCODES["int_tostring"], "random": OPCODES["int_random"],
        "string2int": OPCODES["string_toint"],
    }

    def __init__(self, source: Path) -> None:
        self.source = source
        self.writer = BytecodeWriter()

    def compile(self) -> tuple[bytes, dict[str, dict[str, object]]]:
        try:
            lines = self.source.read_bytes().decode("latin-1").splitlines()
        except OSError as exc:
            raise CompileError(
                f"Failed to read source {self.source} - {exc}",
            ) from exc

        for line_number, line in enumerate(lines, 1):
            self.compile_line(line, line_number)

        if not self.writer.data:
            print("WARNING: Didn't generate any code", file=sys.stderr)

        return self.writer.finish(), self.writer.label_metadata()

    def compile_line(self, line: str, line_number: int) -> None:
        if not line or re.match(r"^\s*#", line):
            return

        match = re.match(r"^\s*:(.*)", line)
        if match:
            self.writer.define_label(match.group(1), line_number)

            return

        for parser in (
            self._store, self._simple, self._jump, self._three_registers,
            self._one_register, self._compare, self._memory, self._stack,
            self._data,
        ):
            if parser(line):
                return

        print(f"WARNING: UNKNOWN LINE: {line}", file=sys.stderr)

    def _store(self, line: str) -> bool:
        match = re.match(r'^\s+store\s+#([0-9]+)\s?,\s?"([^"]*)"', line)
        if match:
            value = encode_string(match.group(2))
            self.writer.emit(
                OPCODES["string_store"],
                parse_number(match.group(1)),
            )
            self.writer.emit_word(len(value), "string length")
            self.writer.data.extend(value)

            return True

        match = re.match(r"^\s+store\s+#([0-9]+)\s?,\s?#([0-9]+)", line)
        if match:
            self.writer.emit(
                OPCODES["reg_store"],
                parse_number(match.group(1)),
                parse_number(match.group(2)),
            )

            return True

        match = re.match(r"^\s+store\s+#([0-9]+)\s?,\s?([^\s]+)", line)
        if match:
            self.writer.emit(OPCODES["int_store"], parse_number(match.group(1)))
            target = match.group(2)

            if is_address(target):
                self.writer.emit_word(parse_number(target), "integer")
            else:
                self.writer.pending.append(
                    PendingAddress(self.writer.offset, target),
                )
                self.writer.data.extend(b"\0\0")

            return True

        return False

    def _simple(self, line: str) -> bool:
        if re.match(r"^\s+exit", line):
            self.writer.emit(OPCODES["exit"])

            return True

        if re.match(r"^\s+nop", line):
            self.writer.emit(OPCODES["nop"])

            return True

        match = re.match(r"^\s*(print_int|print_str|system)\s?#(.*)", line)
        if match:
            opcode = {
                "print_int": "int_print",
                "print_str": "string_print",
                "system": "system",
            }[match.group(1)]

            self.writer.emit(OPCODES[opcode], parse_number(match.group(2)))

            return True

        return False

    def _jump(self, line: str) -> bool:
        match = re.match(r"^\s*(goto|jmp|jmpz|jmpnz|call)\s+([^\s]+)", line)
        if not match:
            return False

        opcode = {
            "goto": "jmp", 
            "jmp": "jmp", 
            "jmpz": "jmpz", 
            "jmpnz": "jmpnz", 
            "call": "call",
        }[match.group(1)]

        self.writer.emit_target(OPCODES[opcode], match.group(2))

        return True

    def _three_registers(self, line: str) -> bool:
        match = re.match(
            r"^\s*(add|and|sub|mul|div|or|xor|concat|memcpy)\s+#([0-9]+)\s*,\s*#([0-9]+)\s*,\s*#([0-9]+)",
            line,
        )
        if not match:
            return False

        if match.group(1) == "memcpy":
            opcode = OPCODES["memcpy"]
        else:
            opcode = self.THREE_REGISTER_OPS[match.group(1)]

        self.writer.emit(
            opcode,
            *(parse_number(value) for value in match.groups()[1:]),
        )

        return True

    def _one_register(self, line: str) -> bool:
        match = re.match(
            r"^\s*(dec|inc|int2string|random|string2int|push|pop|is_string|is_integer)\s+#([0-9]+)",
            line,
        )
        if not match:
            return False

        name = match.group(1)

        if name in self.ONE_REGISTER_OPS:
            opcode = self.ONE_REGISTER_OPS[name]
        else:
            opcode = OPCODES[name]

        self.writer.emit(opcode, parse_number(match.group(2)))

        return True

    def _compare(self, line: str) -> bool:
        match = re.match(
            r"^\s*cmp\s+#([0-9]+)\s*,\s*#([0-9]+)\s*",
            line, re.IGNORECASE,
        )
        if match:
            self.writer.emit(
                OPCODES["cmp_reg"],
                parse_number(match.group(1)),
                parse_number(match.group(2)),
            )

            return True

        match = re.match(r'^\s+cmp\s+#([0-9]+)\s?,\s?"([^"]*)"', line)
        if match:
            value = encode_string(match.group(2))
            self.writer.emit(
                OPCODES["cmp_string"],
                parse_number(match.group(1)),
            )
            self.writer.emit_word(len(value), "string length")
            self.writer.data.extend(value)

            return True

        match = re.match(
            r"^\s*cmp\s+#([0-9]+)\s*,\s*([^\s]+)\s*",
            line, re.IGNORECASE,
        )
        if match:
            self.writer.emit(
                OPCODES["cmp_immediate"],
                parse_number(match.group(1)),
            )
            self.writer.emit_word(parse_number(match.group(2)), "integer")

            return True

        return False

    def _memory(self, line: str) -> bool:
        match = re.match(r"^\s*(peek|poke)\s+#([0-9]+)\s*,\s*#([0-9]+)", line)
        if not match:
            return False

        self.writer.emit(
            OPCODES[match.group(1)],
            parse_number(match.group(2)),
            parse_number(match.group(3)),
        )

        return True

    def _stack(self, line: str) -> bool:
        if re.match(r"^\s*ret\s*", line):
            self.writer.emit(OPCODES["ret"])

            return True

        return False

    def _data(self, line: str) -> bool:
        match = re.match(r"^\s*(?:db|data)\s+(.*)", line, re.IGNORECASE)
        if not match:
            return False

        for value in match.group(1).split(","):
            if value.strip():
                self.writer.emit(parse_number(value.strip()))

        return True


class MetadataWriter:
    FORMAT = "simple-vm-labels"
    VERSION = 1

    @classmethod
    def write(cls, path: Path, bytecode_path: Path, bytecode: bytes,
              labels: dict[str, dict[str, object]]) -> None:
        document = {"format": cls.FORMAT, "version": cls.VERSION,
                    "bytecode": {"file": bytecode_path.name,
                                 "sha256": hashlib.sha256(bytecode).hexdigest(),
                                 "size": len(bytecode)
                                },
                    "labels": labels}

        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile simple.vm assembly to bytecode.",
    )

    parser.add_argument(
        "sources", nargs="+", type=Path,
    )

    parser.add_argument(
        "--labels-json", action="store_true",
        help="write metadata as <output>.labels.json",
    )
    parser.add_argument(
        "--labels-json-file", type=Path, metavar="FILE",
        help="write metadata to FILE (one source only)",
    )

    args = parser.parse_args()
    if args.labels_json_file and len(args.sources) != 1:
        parser.error("--labels-json-file requires exactly one source")

    return args


def main() -> int:
    args = parse_args()
    
    try:
        for source in args.sources:
            bytecode, labels = AssemblyCompiler(source).compile()
            output = source.with_suffix(".raw")
            output.write_bytes(bytecode)

            if args.labels_json or args.labels_json_file:
                MetadataWriter.write(
                    args.labels_json_file or output.with_suffix(".labels.json"),
                    output, bytecode, labels,
                )
    except (CompileError, OSError, ValueError) as exc:
        print(f"compiler: {exc}", file=sys.stderr)

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
