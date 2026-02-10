"""Gemmini Verilog generation via sbt + GemminiGenerator.

This module contains all the logic for configuring and invoking the Chisel
build to produce Gemmini Verilog.  It is used both by the CLI entry-point
(``gemmini-generate``) and programmatically via :meth:`GemminiConfig.generate`.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PRESETS = ["default", "chip", "largeChip", "lean"]

# ---------------------------------------------------------------------------
# Chisel directory — bundled as package data in gemmini_amaranth/chisel/
# ---------------------------------------------------------------------------
CHISEL_DIR = Path(__file__).resolve().parent / "chisel"

# ---------------------------------------------------------------------------
# Chisel dependencies — cloned automatically if missing
# ---------------------------------------------------------------------------
DEPS = {
    "berkeley-hardfloat": {
        "url": "https://github.com/ucb-bar/berkeley-hardfloat.git",
        "commit": "26f00d00c3f3f57480065e02bfcfde3d3b41ec51",
    },
}


def ensure_deps(chisel_dir):
    """Clone Chisel dependencies if missing."""
    for name, info in DEPS.items():
        dep_dir = chisel_dir / name
        if dep_dir.exists() and any(dep_dir.iterdir()):
            continue
        print(f"Cloning {name}...")
        subprocess.run(
            ["git", "clone", info["url"], str(dep_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", info["commit"]],
            cwd=str(dep_dir),
            check=True,
            capture_output=True,
        )
        print(f"  {name} @ {info['commit'][:12]}")


def validate_params(gemmini_args):
    """Validate config constraints before invoking sbt.

    Raises :class:`ValueError` on invalid configuration.
    """
    errors = []

    mesh_rows = int(gemmini_args.get("meshRows", "16"))
    mesh_cols = int(gemmini_args.get("meshColumns", "16"))
    tile_rows = int(gemmini_args.get("tileRows", "1"))
    tile_cols = int(gemmini_args.get("tileColumns", "1"))

    block_rows = mesh_rows * tile_rows
    block_cols = mesh_cols * tile_cols

    if block_rows != block_cols:
        errors.append(
            f"Array must be square: mesh_rows*tile_rows ({block_rows}) != "
            f"mesh_columns*tile_columns ({block_cols})"
        )

    dim = block_rows
    if dim < 2:
        errors.append(f"Array dimension must be >= 2, got {dim}")
    if dim & (dim - 1) != 0:
        errors.append(f"Array dimension must be power of 2, got {dim}")

    input_width = int(gemmini_args.get("inputWidth", "8"))
    sp_banks = int(gemmini_args.get("spBanks", "4"))
    acc_banks = int(gemmini_args.get("accBanks", "2"))
    sp_kb = int(gemmini_args.get("spCapacityKB", "256"))
    acc_kb = int(gemmini_args.get("accCapacityKB", "64"))

    sp_width = block_cols * input_width
    sp_bank_entries = sp_kb * 1024 * 8 // (sp_banks * sp_width)
    if sp_bank_entries == 0:
        errors.append(f"Scratchpad too small: {sp_kb}KB / {sp_banks} banks / {sp_width}-bit rows = 0 entries")
    elif sp_bank_entries & (sp_bank_entries - 1) != 0:
        errors.append(f"SP bank entries must be power of 2, got {sp_bank_entries}")
    elif sp_bank_entries % dim != 0:
        errors.append(f"SP bank entries ({sp_bank_entries}) must be divisible by array dim ({dim})")

    acc_width = int(gemmini_args.get("accWidth", "32"))
    acc_row_width = block_cols * acc_width
    acc_bank_entries = acc_kb * 1024 * 8 // (acc_banks * acc_row_width)
    if acc_bank_entries == 0:
        errors.append(f"Accumulator too small: {acc_kb}KB / {acc_banks} banks / {acc_row_width}-bit rows = 0 entries")
    elif acc_bank_entries % dim != 0:
        errors.append(f"ACC bank entries ({acc_bank_entries}) must be divisible by array dim ({dim})")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def config_hash(gen_params):
    """Return a 12-char hex hash of the generation parameters."""
    canonical = json.dumps(gen_params, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# Maps user-facing kwargs to Chisel --gemmini-key=value args.
def _snake_to_camel(name):
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# Config fields that map directly to --gemmini-key=value Chisel args.
_GEN_FIELDS = [
    "mesh_rows", "mesh_columns", "tile_rows", "tile_columns",
    "dataflow", "sp_banks", "acc_banks", "input_width", "acc_width",
    "dma_buswidth", "dma_maxbytes", "max_in_flight_mem_reqs",
]

# Properties that map to Chisel args (not dataclass fields).
_GEN_PROPS = {
    "sp_capacity_kb": "spCapacityKB",
    "acc_capacity_kb": "accCapacityKB",
}

# Boolean flags (True → "false" because Chisel defaults are true).
_GEN_FLAGS = {
    "no_training_convs": "hasTrainingConvs",
    "no_max_pool": "hasMaxPool",
    "no_nonlinear_activations": "hasNonlinearActivations",
}


def build_gen_params(config, preset=None):
    """Build the ``--gemmini-key=value`` args dict from a GemminiConfig."""
    gemmini_args = {}

    if preset:
        gemmini_args["preset"] = preset

    for field in _GEN_FIELDS:
        val = getattr(config, field)
        if val is not None:
            gemmini_args[_snake_to_camel(field)] = str(val)

    for prop, key in _GEN_PROPS.items():
        gemmini_args[key] = str(getattr(config, prop))

    for flag, key in _GEN_FLAGS.items():
        if getattr(config, flag):
            gemmini_args[key] = "false"

    return gemmini_args


def build_sbt_command(gemmini_args, output_dir):
    """Build the full sbt runMain command string."""
    parts = ["GemminiGenerator"]
    for key, val in sorted(gemmini_args.items()):
        parts.append(f"--gemmini-{key}={val}")

    abs_output = str(Path(output_dir).resolve())
    parts.append("--target-dir")
    parts.append(abs_output)

    inner = " ".join(parts)
    return f"""sbt 'runMain {inner}'"""


def run_sbt(sbt_cmd, chisel_dir, verbose=False):
    """Execute the sbt command in the chisel directory.

    Raises :class:`subprocess.CalledProcessError` on failure.
    """
    result = subprocess.run(
        sbt_cmd,
        shell=True,
        cwd=str(chisel_dir),
        capture_output=not verbose,
        text=True,
    )

    if result.returncode != 0:
        msg = "sbt command failed!"
        if not verbose and result.stderr:
            msg += "\n" + result.stderr
        if not verbose and result.stdout:
            lines = result.stdout.strip().split("\n")
            msg += "\n" + "\n".join(lines[-30:])
        raise subprocess.CalledProcessError(result.returncode, sbt_cmd, msg)


def augment_config_json(gemmini_args, output_dir):
    """Augment the Scala-generated gemmini_config.json with Python-side metadata."""
    config_path = Path(output_dir) / "gemmini_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    config["generator"] = "GemminiGenerator"
    config["cliParameters"] = gemmini_args
    config["outputDir"] = str(output_dir)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config JSON: {config_path}")


def generate_verilog(gen_params, output_dir, verbose=False):
    """Orchestrate Verilog generation: ensure deps, validate, run sbt, augment JSON."""
    ensure_deps(CHISEL_DIR)
    validate_params(gen_params)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sbt_cmd = build_sbt_command(gen_params, output_dir)

    if verbose:
        print(f"  Command: {sbt_cmd}")
        print(f"  Output dir: {output_dir}")
        print(f"  Parameters: {json.dumps(gen_params, indent=2)}")

    print("Generating Gemmini Verilog...")
    run_sbt(sbt_cmd, CHISEL_DIR, verbose)

    # Verify output files
    verilog_file = output_dir / "Gemmini.v"
    header_file = output_dir / "gemmini_params.h"
    config_file = output_dir / "gemmini_config.json"

    missing = []
    if not verilog_file.exists():
        v_files = list(output_dir.glob("*.v")) + list(output_dir.glob("*.sv"))
        if not v_files:
            missing.append(str(verilog_file))
        else:
            verilog_file = v_files[0]
            print(f"Verilog output: {verilog_file}")

    if not header_file.exists():
        h_files = list(output_dir.glob("gemmini_params*.h"))
        if not h_files:
            missing.append(str(header_file))
        else:
            header_file = h_files[0]

    if not config_file.exists():
        missing.append(str(config_file))

    if missing:
        print(f"WARNING: Expected output files not found: {missing}", file=sys.stderr)
    else:
        print(f"Generated Verilog: {verilog_file} ({verilog_file.stat().st_size:,} bytes)")
        print(f"Generated header:  {header_file} ({header_file.stat().st_size:,} bytes)")
        print(f"Generated config:  {config_file} ({config_file.stat().st_size:,} bytes)")

    augment_config_json(gen_params, output_dir)
    print("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate configurable Gemmini Verilog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
    gemmini-generate                                          # defaults: int8, 16x16
    gemmini-generate --preset chip                            # named preset
    gemmini-generate --mesh-rows 8 --mesh-columns 8
    gemmini-generate --dry-run                                # show sbt command only
""",
    )
    p.add_argument("--preset", choices=PRESETS, help="Named preset config")
    p.add_argument("--mesh-rows", type=int, default=None)
    p.add_argument("--mesh-columns", type=int, default=None)
    p.add_argument("--tile-rows", type=int, default=None)
    p.add_argument("--tile-columns", type=int, default=None)
    p.add_argument("--dataflow", choices=["OS", "WS", "BOTH"], default=None)
    p.add_argument("--sp-banks", type=int, default=None)
    p.add_argument("--sp-bank-entries", type=int, default=None)
    p.add_argument("--acc-banks", type=int, default=None)
    p.add_argument("--acc-bank-entries", type=int, default=None)
    p.add_argument("--dma-maxbytes", type=int, default=None)
    p.add_argument("--dma-buswidth", type=int, default=None)
    p.add_argument("--max-in-flight-mem-reqs", type=int, default=None)
    p.add_argument("--no-training-convs", action="store_true")
    p.add_argument("--no-max-pool", action="store_true")
    p.add_argument("--no-nonlinear-activations", action="store_true")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: build/<config-hash>)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def cli_main(argv=None):
    """CLI entry point for Gemmini Verilog generation."""
    args = _parse_args(argv)

    # Build GemminiConfig from CLI args, then gen_params from that
    from gemmini_amaranth.gemmini import GemminiConfig
    from dataclasses import fields
    cfg_overrides = {}
    for f in fields(GemminiConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            cfg_overrides[f.name] = val
    config = GemminiConfig(**cfg_overrides)

    gen_params = build_gen_params(config, preset=args.preset)
    validate_params(gen_params)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (Path.cwd() / "build" / config_hash(gen_params)).resolve()

    sbt_cmd = build_sbt_command(gen_params, output_dir)

    if args.dry_run:
        print("Dry run - would execute:")
        print(f"  cd {CHISEL_DIR}")
        print(f"  {sbt_cmd}")
        print(f"  output: {output_dir}")
        return

    generate_verilog(gen_params, output_dir, args.verbose)
