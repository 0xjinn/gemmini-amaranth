import os
import inspect
import json
import random
from dataclasses import asdict, make_dataclass, field
from pathlib import Path
import hashlib
import pickle

import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, ReadWrite, Timer
from cocotb.handle import Immediate
from cocotb.queue import Queue
from cocotb_tools.runner import get_runner

from gemmini_amaranth.bus import RespType

from amaranth.back.verilog import convert


def config_from_component(cls):
    """Auto-generate a frozen config dataclass from a Component's __init__ signature."""
    sig = inspect.signature(cls.__init__)
    fields = []
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.default is inspect.Parameter.empty:
            fields.append((name, int))
        else:
            fields.append((name, int, field(default=param.default)))
    config_cls = make_dataclass(f"{cls.__name__}Config", fields, frozen=True)
    config_cls.fromdict = classmethod(lambda cls, d: cls(**d))
    return config_cls


async def cocotb_init(dut, config_cls=None):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))

    dut.rst.set(Immediate(1))
    await Timer(100, unit="ns")
    dut.rst.set(Immediate(0))

    if config_cls is None:
        return None

    with open(Path(cocotb.plusargs.get("config_dir")) / "config.json") as f:
        config = config_cls.fromdict(json.load(f))
    return config


def cocotb_run(dut, config=None, module=None, sim="verilator", testfilter=None, always=False, waves=False, timescale=("1ns", "1ps"),
    test_args=[], build_args=[], logtest=False, stdout_loglevel="WARNING", build_dir=None):
    os.environ["GPI_LOG_LEVEL"] = os.getenv("GPI_LOG_LEVEL", stdout_loglevel)
    os.environ["COCOTB_LOG_LEVEL"] = os.getenv("COCOTB_LOG_LEVEL", stdout_loglevel)

    top = dut.__class__.__name__
    build_dir = build_dir if build_dir else Path(__file__).parent / "build"

    if config is not None:
        confhash = hashlib.md5(pickle.dumps(config)).hexdigest()[:8]
        config_dir = build_dir / f"{top}_{confhash}"
    else:
        config_dir = build_dir / top
    config_dir.mkdir(exist_ok=True, parents=True)

    src = config_dir / f"{top}.v"

    if not src.exists() or always:
        with open(src, "w") as f:
            f.write(convert(dut, name=top, emit_src=False))

    if testfilter is not None:
        os.environ["COCOTB_TEST_FILTER"] = testfilter

    if sim == "verilator":
        build_args += ["--Wno-fatal"]
        if not os.getenv("CI"):
            build_args += ["--quiet"]
    if waves and sim == "verilator":
        build_args += ["--trace-fst"]

    if config is not None:
        with open(config_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=4)
        test_args += [f"+config_dir={config_dir}"]

    module = inspect.getmodule(inspect.stack()[1].frame).__name__ if module is None else module
    runner = get_runner(sim)

    sim_dir = config_dir / "sim_build"

    srcs = [src]
    if config is not None and hasattr(config, "get_verilog_sources"):
        srcs.extend(config.get_verilog_sources())

    runner.build(sources=srcs, hdl_toplevel=top, build_args=build_args, always=always, waves=waves, build_dir=sim_dir,
        timescale=timescale, log_file=sim_dir / "build.log")
    runner.test(hdl_toplevel=top, test_module=module, waves=waves, test_args=test_args, build_dir=sim_dir, test_dir=sim_dir,
        log_file=sim_dir / "test.log" if logtest else None)


# stream -> queue
async def stream_consumer(clk, queue, ready, valid, payload, rand=False):
    while True:
        await RisingEdge(clk)
        ready.value = random.getrandbits(1) if rand else 1
        await ReadOnly()
        if ready.value and valid.value:
            await queue.put({k: v.value for k, v in payload.items()})

# queue -> stream, waits for ready
async def stream_producer(clk, queue, ready, valid, payload, timeout=1000, rand=False):
    while True:
        await RisingEdge(clk)
        valid.value = 0
        p = random.getrandbits(1) if rand else 1
        if p and queue.qsize():
            valid.value = 1
            for k, v in (await queue.get()).items():
                payload[k].value = v
            await ReadWrite()
            wait_cycles = 0
            while ready.value == 0:
                await RisingEdge(clk)
                await ReadWrite()
                wait_cycles += 1
                assert wait_cycles < timeout, "timeout waiting ready"


def packed(byte_list, src_bits=8):
    val = 0
    for i, b in enumerate(byte_list):
        val |= (b & ((1 << src_bits) - 1)) << (i * src_bits)
    return val


def unpacked(val, total_bits, chunk_bits):
    mask = (1 << chunk_bits) - 1
    return [(val >> (i * chunk_bits)) & mask for i in range(total_bits // chunk_bits)]


class GemminiAxiInterface:
    def __init__(self, dut, mem_size=4 * 1024 * 1024, rand=False):
        self.dut = dut
        self.memory = np.zeros(mem_size, dtype=np.uint8)
        self.rand = rand

        self.producer_intfs = {
            "ctrl__aw": {"queue": Queue(), "payload": ["addr", "prot"]},
            "ctrl__w": {"queue": Queue(), "payload": ["data", "strb"]},
            "ctrl__ar": {"queue": Queue(), "payload": ["addr", "prot"]},
            "bus__r": {"queue": Queue(), "payload": ["resp", "data", "last"]},
            "bus__b": {"queue": Queue(), "payload": ["resp"]},
        }

        self.consumer_intfs = {
            "ctrl__r": {"queue": Queue(), "payload": ["data", "resp"]},
            "ctrl__b": {"queue": Queue(), "payload": ["resp"]},
            "bus__ar": {"queue": Queue(), "payload": ["addr", "len", "size"]},
            "bus__aw": {"queue": Queue(), "payload": ["addr", "len", "size"]},
            "bus__w": {"queue": Queue(), "payload": ["data", "last"]},
        }

    async def init(self):
        cocotb.start_soon(self.memory_read_process())
        cocotb.start_soon(self.memory_write_process())

        for k, v in self.producer_intfs.items():
            cocotb.start_soon(stream_producer(self.dut.clk, v["queue"], **self._stream_payload(k, v["payload"]), rand=self.rand))

        for k, v in self.consumer_intfs.items():
            cocotb.start_soon(stream_consumer(self.dut.clk, v["queue"], **self._stream_payload(k, v["payload"]), rand=self.rand))

    async def read_csr(self, addr):
        await self.producer_intfs["ctrl__ar"]["queue"].put({"addr": addr, "prot": 0})
        return (await self.consumer_intfs["ctrl__r"]["queue"].get())["data"]

    async def write_csr(self, addr, data):
        await self.producer_intfs["ctrl__aw"]["queue"].put({"addr": addr, "prot": 0})
        await self.producer_intfs["ctrl__w"]["queue"].put({"data": data, "strb": 0xF})
        assert (await self.consumer_intfs["ctrl__b"]["queue"].get())["resp"] == RespType.OKAY.value

    async def memory_read_process(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.consumer_intfs["bus__ar"]["queue"].qsize():
                req = await self.consumer_intfs["bus__ar"]["queue"].get()
                addr = req["addr"].to_unsigned()
                size = 1 << req["size"].to_unsigned()
                length = req["len"].to_unsigned()
                assert addr >= 0 and addr + size * (length + 1) <= len(self.memory)
                for i in range(0, length + 1):
                    await self.producer_intfs["bus__r"]["queue"].put({
                        "resp": RespType.OKAY.value,
                        "data": packed(self.memory[addr + size * i:addr + size * (i + 1)].tolist(), 8),
                        "last": int(i == length),
                    })

    async def memory_write_process(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.consumer_intfs["bus__aw"]["queue"].qsize() and self.consumer_intfs["bus__w"]["queue"].qsize():
                req = await self.consumer_intfs["bus__aw"]["queue"].get()
                addr = req["addr"].to_unsigned()
                size = 1 << req["size"].to_unsigned()
                length = req["len"].to_unsigned()
                assert addr >= 0 and addr + size * (length + 1) <= len(self.memory)

                data = []
                while True:
                    wdata = await self.consumer_intfs["bus__w"]["queue"].get()
                    data.extend(unpacked(wdata["data"].to_unsigned(), size * 8, 8))
                    if wdata["last"]:
                        break
                self.memory[addr:addr + size * (length + 1)] = data
                await self.producer_intfs["bus__b"]["queue"].put({"resp": RespType.OKAY.value})

    def _stream_payload(self, name, payload):
        hs_sigs = {k: getattr(self.dut, name + k) for k in ["ready", "valid"]}
        return {"payload": {sn: getattr(self.dut, name + sn) for sn in payload}, **hs_sigs}
