"""Cocotb tests: Gemmini through AXI4 wrapper — matmul + performance counters."""

import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, ReadOnly, ReadWrite
from cocotb.queue import Queue

from gemmini_amaranth.gemmini import Gemmini, GemminiConfig
from gemmini_amaranth.driver import GemminiISA, sp_addr, acc_addr, GARBAGE_ADDR, CounterEvent, CounterExternal
from test.helpers import cocotb_run, cocotb_init, GemminiAxiInterface

DIM = 16
ADDR_A, ADDR_B, ADDR_OUT = 0x1000, 0x2000, 0x4000
_MASK64 = (1 << 64) - 1
ALWAYS = int(os.getenv("ALWAYS", 0))
BUILD_DIR = str(Path(__file__).parent / ".." / "build")

# CMD_LAYOUT: funct[7] | rs1[64] | rs2[64] | xd[1] | rd[5] (LSB-first, 141 bits)
def _pack_cmd(funct, rs1, rs2, xd=0, rd=0):
    return (funct & 0x7F) | ((rs1 & _MASK64) << 7) | ((rs2 & _MASK64) << 71) | ((xd & 1) << 135) | ((rd & 0x1F) << 136)

# RESP_LAYOUT: rd[5] | data[64] (LSB-first, 69 bits)
def _unpack_resp(val): return {"rd": val & 0x1F, "data": (val >> 5) & _MASK64}

async def send_cmd(dut, funct, rs1, rs2, xd=0, rd=0):
    await RisingEdge(dut.clk)
    dut.cmd__payload.value = _pack_cmd(funct, rs1, rs2, xd, rd)
    dut.cmd__valid.value = 1
    await ReadWrite()
    while dut.cmd__ready.value == 0:
        await RisingEdge(dut.clk)
        await ReadWrite()
    await RisingEdge(dut.clk)
    dut.cmd__valid.value = 0

async def resp_sink(dut, queue):
    while True:
        await RisingEdge(dut.clk)
        dut.resp__ready.value = 1
        await ReadOnly()
        if dut.resp__ready.value and dut.resp__valid.value:
            queue.put_nowait(_unpack_resp(int(dut.resp__payload.value)))

async def wait_idle(axi, timeout=50000):
    for _ in range(timeout):
        await RisingEdge(axi.dut.clk)
        status = await axi.read_csr(GemminiConfig.csr_offsets["status"])
        if not (status.to_unsigned() & 1): return
    raise TimeoutError(f"Gemmini still busy after {timeout} cycles")

def _load(mem, addr, mat, dtype=np.int8):
    flat = np.ascontiguousarray(mat, dtype=dtype).view(np.uint8).ravel()
    mem[addr:addr + len(flat)] = flat

async def setup(dut, with_resp=False, rand=False):
    await cocotb_init(dut, GemminiConfig)
    axi = GemminiAxiInterface(dut, rand=rand)
    await axi.init()
    dut.resp__ready.value = 1
    if with_resp:
        resp_q = Queue()
        cocotb.start_soon(resp_sink(dut, resp_q))
        return axi, resp_q
    return axi

async def run_matmul(dut, axi, A=None, B=None):
    rng = np.random.default_rng(123)
    if A is None: A = rng.integers(-5, 5, size=(DIM, DIM), dtype=np.int8)
    if B is None: B = rng.integers(-5, 5, size=(DIM, DIM), dtype=np.int8)
    _load(axi.memory, ADDR_A, A)
    _load(axi.memory, ADDR_B, B)
    instrs = [
        GemminiISA.config_ex(dataflow=1, acc_scale=0x3F800000),
        GemminiISA.config_ld(stride=DIM, scale=0x3F800000),
        GemminiISA.mvin(ADDR_A, sp_addr(0), DIM, DIM),
        GemminiISA.mvin(ADDR_B, sp_addr(DIM), DIM, DIM),
        GemminiISA.preload(sp_addr(DIM), acc_addr(0), DIM, DIM, DIM, DIM),
        GemminiISA.compute_preloaded(sp_addr(0), GARBAGE_ADDR, a_cols=DIM, a_rows=DIM, bd_cols=DIM, bd_rows=DIM),
        GemminiISA.config_st(stride=DIM * 4, acc_scale=0x3F800000),
        GemminiISA.mvout(ADDR_OUT, acc_addr(0, read_full=True), DIM, DIM),
    ]
    for f, rs1, rs2 in instrs: await send_cmd(dut, f, rs1, rs2)
    await wait_idle(axi)
    return np.frombuffer(bytes(axi.memory[ADDR_OUT:ADDR_OUT + DIM * DIM * 4]),
                         dtype=np.int32).reshape(DIM, DIM)

@cocotb.test()
async def test_matmul_identity(dut):
    axi = await setup(dut)
    A, B = np.eye(DIM, dtype=np.int8), np.arange(DIM * DIM, dtype=np.int8).reshape(DIM, DIM)
    result = await run_matmul(dut, axi, A, B)
    expected = A.astype(np.int32) @ B.astype(np.int32)
    dut._log.info(f"Expected[0]: {expected[0]}\nGot[0]:      {result[0]}")
    np.testing.assert_array_equal(result, expected)
    dut._log.info("PASS: identity matmul")

@cocotb.test()
async def test_matmul_random(dut):
    axi = await setup(dut, rand=True)
    rng = np.random.default_rng(42)
    A = rng.integers(-10, 10, size=(DIM, DIM), dtype=np.int8)
    B = rng.integers(-10, 10, size=(DIM, DIM), dtype=np.int8)
    result = await run_matmul(dut, axi, A, B)
    expected = A.astype(np.int32) @ B.astype(np.int32)
    dut._log.info(f"Expected[0]: {expected[0]}\nGot[0]:      {result[0]}")
    np.testing.assert_array_equal(result, expected)
    dut._log.info("PASS: random matmul")

@cocotb.test()
async def test_matmul_tiled(dut):
    axi = await setup(dut)
    N = 2 * DIM
    rng = np.random.default_rng(99)
    A = rng.integers(-5, 5, size=(N, N), dtype=np.int8)
    B = rng.integers(-5, 5, size=(N, N), dtype=np.int8)
    _load(axi.memory, ADDR_A, A)
    _load(axi.memory, ADDR_B, B)

    instrs = [GemminiISA.config_ex(dataflow=1, acc_scale=0x3F800000)]
    n_tiles = N // DIM
    for ti in range(n_tiles):
        for tj in range(n_tiles):
            for tk in range(n_tiles):
                instrs += [GemminiISA.config_ld(stride=N, scale=0x3F800000),
                           GemminiISA.mvin(ADDR_A + ti * DIM * N + tk * DIM, sp_addr(0), DIM, DIM),
                           GemminiISA.config_ld(stride=N, scale=0x3F800000),
                           GemminiISA.mvin(ADDR_B + tk * DIM * N + tj * DIM, sp_addr(DIM), DIM, DIM)]
                a_acc = acc_addr(0) if tk == 0 else acc_addr(0, accumulate=True)
                instrs += [GemminiISA.preload(sp_addr(DIM), a_acc, DIM, DIM, DIM, DIM),
                           GemminiISA.compute_preloaded(sp_addr(0), GARBAGE_ADDR,
                               a_cols=DIM, a_rows=DIM, bd_cols=DIM, bd_rows=DIM)]
            instrs += [GemminiISA.config_st(stride=N * 4, acc_scale=0x3F800000),
                       GemminiISA.mvout(ADDR_OUT + ti * DIM * N * 4 + tj * DIM * 4,
                                        acc_addr(0, read_full=True), DIM, DIM)]

    for f, rs1, rs2 in instrs: await send_cmd(dut, f, rs1, rs2)
    await wait_idle(axi)
    result = np.frombuffer(bytes(axi.memory[ADDR_OUT:ADDR_OUT + N * N * 4]),
                           dtype=np.int32).reshape(N, N)
    expected = A.astype(np.int32) @ B.astype(np.int32)
    dut._log.info(f"Expected[0]: {expected[0]}\nGot[0]:      {result[0]}")
    np.testing.assert_array_equal(result, expected)
    dut._log.info("PASS: tiled 32x32 matmul")

# counter helpers
async def _ccmd(dut, funct, rs1, rs2, rq):
    await send_cmd(dut, funct, rs1, rs2, xd=1, rd=0)
    return await rq.get()

async def _ccfg(dut, idx, ev, rq, ext=False):
    await _ccmd(dut, *GemminiISA.counter_configure(idx, ev, external=ext), rq)
async def _creset(dut, rq): await _ccmd(dut, *GemminiISA.counter_reset(), rq)
async def _cread(dut, idx, rq): return (await _ccmd(dut, *GemminiISA.counter_read(idx), rq))["data"]
async def _csnap(dut, rq): await _ccmd(dut, *GemminiISA.counter_snapshot(), rq)
async def _csnap_reset(dut, rq): await _ccmd(dut, *GemminiISA.counter_snapshot_reset(), rq)

@cocotb.test()
async def test_perf_counters(dut):
    axi, rq = await setup(dut, with_resp=True)
    expected_nonzero = {1, 2, 3, 9, 12, 18, 21, 24, 35, 42}
    expected_nonzero_ext = {4, 5}

    for round_idx in range(6):
        events = [(i, min(round_idx * 8 + i, CounterEvent.N - 1)
                   if round_idx * 8 + i < CounterEvent.N else 0, False) for i in range(8)]
        for idx, ev, ext in events: await _ccfg(dut, idx, ev, rq, ext=ext)
        await _creset(dut, rq)
        await run_matmul(dut, axi)
        for idx, ev, _ in events:
            if ev == 0: continue
            val = await _cread(dut, idx, rq)
            if ev in CounterEvent.STALE:
                assert val == 0, f"stale event {ev} should be 0, got {val}"
                dut._log.info(f"  event {ev:2d} (stale): {val} == 0 OK")
            elif ev in expected_nonzero:
                assert val > 0, f"event {ev} expected > 0, got {val}"
                dut._log.info(f"  event {ev:2d}: {val} > 0 OK")
            else: dut._log.info(f"  event {ev:2d}: {val}")

    ext_events = list(range(1, CounterExternal.N))
    for i, ev in enumerate(ext_events[:8]): await _ccfg(dut, i, ev, rq, ext=True)
    await _creset(dut, rq)
    await run_matmul(dut, axi)
    for i, ev in enumerate(ext_events[:8]):
        val = await _cread(dut, i, rq)
        if ev in expected_nonzero_ext:
            assert val > 0, f"ext event {ev} expected > 0, got {val}"
            dut._log.info(f"  ext {ev}: {val} > 0 OK")
        else: dut._log.info(f"  ext {ev}: {val}")
    dut._log.info("PASS: perf counters verified")

@cocotb.test()
async def test_stale_counters_zero(dut):
    axi, rq = await setup(dut, with_resp=True)
    stale = sorted(CounterEvent.STALE)
    for i, ev in enumerate(stale): await _ccfg(dut, i, ev, rq)
    await _creset(dut, rq)
    await run_matmul(dut, axi)
    for i, ev in enumerate(stale):
        val = await _cread(dut, i, rq)
        assert val == 0, f"stale event {ev} should be 0, got {val}"
        dut._log.info(f"  stale event {ev}: {val} == 0 OK")
    dut._log.info("PASS: all stale counters read 0")

@cocotb.test()
async def test_counter_snapshot(dut):
    axi, rq = await setup(dut, with_resp=True)
    await _ccfg(dut, 0, CounterEvent.RESERVATION_STATION_ACTIVE_CYCLES, rq)
    await _creset(dut, rq)
    await run_matmul(dut, axi)
    await _csnap(dut, rq)
    val1 = await _cread(dut, 0, rq)
    assert val1 > 0, f"RS_ACTIVE should be > 0, got {val1}"
    for _ in range(100): await RisingEdge(dut.clk)
    val2 = await _cread(dut, 0, rq)
    assert val1 == val2, f"snapshot values should match: {val1} != {val2}"
    dut._log.info(f"  snapshot: {val1} (consistent)")
    await _csnap_reset(dut, rq)
    val3 = await _cread(dut, 0, rq)
    assert val3 >= val1, f"live {val3} should be >= snapshot {val1}"
    dut._log.info(f"  live after clear: {val3}")
    dut._log.info("PASS: counter snapshot works")


def test_gemmini():
    config = GemminiConfig(verilog_dir=BUILD_DIR)
    dut = Gemmini(config)
    cocotb_run(dut, config, always=ALWAYS)
