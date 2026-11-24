#!/usr/bin/env python3
"""Decompiler for simple.vm bytecode with optional label restoration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


class DecompileError(Exception):
    """Raised when bytecode or its label metadata is malformed."""


class LabelMetadata:
    """Loads and validates the versioned sidecar file produced by compiler.py."""

    FORMAT = "simple-vm-labels"
    VERSION = 1

    def __init__(self, labels: dict[int, str]) -> None:
        self.labels = labels

    @classmethod
    def load(cls, path: Path, bytecode: bytes) -> "LabelMetadata":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DecompileError(f"cannot read label metadata {path}: {exc}") from exc

        format_differs = document.get("format") != cls.FORMAT
        version_differs = document.get("version") != cls.VERSION
        if format_differs or version_differs:
            raise DecompileError("unsupported label metadata format")

        info = document.get("bytecode", {})
        digest = hashlib.sha256(bytecode).hexdigest()

        hash_differs = info.get("sha256") != digest
        size_differs = info.get("size") != len(bytecode)
        if hash_differs or size_differs:
            raise DecompileError("label metadata does not match this bytecode")

        labels: dict[int, str] = {}
        for address, entry in document.get("labels", {}).items():
            try:
                offset = int(address)
            except (TypeError, ValueError) as exc:
                raise DecompileError(f"invalid label address: {address}") from exc

            label_is_in_range = 0 <= offset <= len(bytecode)
            entry_is_dict = isinstance(entry, dict)
            name_is_str = isinstance(entry.get("name"), str)
            if not label_is_in_range or not entry_is_dict or not name_is_str:
                raise DecompileError(f"invalid label at address {address}")

            labels[offset] = entry["name"]

        return cls(labels)


class BytecodeReader:
    """Bounds-checked sequential reader for a bytecode buffer."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    @property
    def at_end(self) -> bool:
        return self.position >= len(self.data)

    def byte(self) -> int:
        return int.from_bytes(self.bytes(1))

    def word(self) -> int:
        return int.from_bytes(self.bytes(2), byteorder='little')

    def bytes(self, count: int) -> bytes:
        self._need(count)
        value = self.data[self.position:self.position + count]
        self.position += count

        return value

    def _need(self, count: int) -> None:
        if self.position + count > len(self.data):
            raise DecompileError(f"truncated instruction at 0x{self.position:04X}")


class BytecodeDecompiler:
    """Renders bytecode into source accepted by the legacy Perl compiler."""

    THREE_REGISTER = {
        0x20: "xor", 0x21: "add", 0x22: "sub",
        0x23: "mul", 0x24: "div", 0x27: "and",
        0x28: "or", 0x32: "concat", 0x62: "memcpy",
    }

    ONE_REGISTER = {
        0x02: "print_int", 0x03: "int2string", 0x04: "random", 0x25: "inc", 
        0x26: "dec", 0x31: "print_str", 0x33: "system", 0x34: "string2int", 
        0x43: "is_string", 0x44: "is_integer", 0x70: "push", 0x71: "pop",
    }

    JUMPS = {
        0x10: "jmp", 0x11: "jmpz", 0x12: "jmpnz", 0x73: "call",
    }

    def __init__(
        self, data: bytes, 
        labels: LabelMetadata | None = None, 
        show_address: bool = False,
    ) -> None:
        self.reader = BytecodeReader(data)
        self.labels = labels.labels if labels else {}
        self.show_address = show_address

    def decompile(self) -> str:
        lines: list[str] = []
        while not self.reader.at_end:
            start = self.reader.position

            if start in self.labels:
                lines.append(self._label_line(start))

            opcode = self.reader.byte()

            lines.append(self._instruction_line(start, opcode))

        if self.reader.position in self.labels:
            lines.append(self._label_line(self.reader.position))

        return "".join(lines)

    def _instruction_line(self, start: int, opcode: int) -> str:
        if opcode == 0x00:
            return self._line(start, "exit")

        if opcode == 0x01:
            register, value = self.reader.byte(), self.reader.word()

            return self._line(start, f"store #{register}, 0x{value:04X}")

        if opcode in self.ONE_REGISTER:
            return self._line(
                start, f"{self.ONE_REGISTER[opcode]} #{self.reader.byte()}",
            )

        if opcode in self.JUMPS:
            target = self.reader.word()

            return self._line(
                start,
                f"{self.JUMPS[opcode]} {self.labels.get(target, f'0x{target:04X}')}",
            )

        if opcode in self.THREE_REGISTER:
            first = self.reader.byte()
            second = self.reader.byte()
            third = self.reader.byte()

            return self._line(
                start,
                f"{self.THREE_REGISTER[opcode]} #{first}, #{second}, #{third}",
            )

        if opcode in (0x30, 0x42):
            register = self.reader.byte()
            length = self.reader.word()
            value = self._quote(self.reader.bytes(length))
            mnemonic = "store" if opcode == 0x30 else "cmp"

            return self._line(start, f'{mnemonic} #{register}, "{value}"')

        if opcode == 0x40:
            first = self.reader.byte()
            second = self.reader.byte()

            return self._line(start, f"cmp #{first},#{second}")

        if opcode == 0x41:
            register = self.reader.byte()
            value = self.reader.word()

            return self._line(start, f"cmp #{register},0x{value:04X}")

        if opcode == 0x50:
            return self._line(start, "nop")

        if opcode == 0x51:
            destination = self.reader.byte()
            source = self.reader.byte()

            return self._line(start, f"store #{destination},#{source}")

        if opcode in (0x60, 0x61):
            first = self.reader.byte()
            second = self.reader.byte()

            return self._line(
                start,
                f"{'peek' if opcode == 0x60 else 'poke'} #{first}, #{second}",
            )

        if opcode == 0x72:
            return self._line(start, "ret")

        return self._line(start, f"DATA {opcode}")

    def _line(self, offset: int, text: str) -> str:
        return (f"{offset:04X}" if self.show_address else "") + f"\t{text}\n"

    def _label_line(self, offset: int) -> str:
        return (f"{offset:04X}" if self.show_address else "") + \
            f":{self.labels[offset]}\n"

    @staticmethod
    def _quote(value: bytes) -> str:
        # The old compiler accepts these exact two escapes.
        return value.decode("latin-1").replace("\n", r"\n").replace("\t", r"\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompile simple.vm bytecode.",
    )

    parser.add_argument(
        "files", nargs="+", type=Path,
    )

    parser.add_argument(
        "--show-address", action="store_true",
    )
    parser.add_argument(
        "--labels-json", type=Path, metavar="FILE",
        help="restore labels from compiler metadata",
    )

    args = parser.parse_args()
    if args.labels_json and len(args.files) != 1:
        parser.error("--labels-json requires exactly one bytecode file")

    return args


def main() -> int:
    args = parse_args()

    try:
        for path in args.files:
            bytecode = path.read_bytes()
            labels = LabelMetadata.load(args.labels_json, bytecode) \
                if args.labels_json else None

            sys.stdout.write(
                BytecodeDecompiler(bytecode, labels, args.show_address).decompile()
            )
    except (OSError, DecompileError) as exc:
        print(f"decompiler: {exc}", file=sys.stderr)

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
