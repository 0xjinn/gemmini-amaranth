import struct, math, random
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, ReadWrite, Timer
from cocotb.queue import Queue

_MASK64 = (1 << 64) - 1

# payload pack/unpack for Amaranth-wrapped Gemmini
def pack_rocc_cmd(funct, rs1, rs2, xd=0, rd=0):
    inst = (funct & 0x7F) | ((xd & 1) << 17) | (1 << 18) | (1 << 19) | ((rd & 0x1F) << 20) | (0x0B << 25)
    status = (3 << 35) | (3 << 38) | (3 << 92)
    return inst | ((rs1 & _MASK64) << 32) | ((rs2 & _MASK64) << 96) | (status << 160)

def unpack_read_req(val):
    return {"lg_size": val & 0x7, "bytes_read": (val >> 3) & 0x7F, "shift": (val >> 10) & 0x3F,
            "vaddr": (val >> 16) & 0xFFFFFFFF, "source": (val >> 48) & 0xF}

def pack_read_resp(source, data, lg_size, last):
    return ((source & 0xF) | ((int(data) & ((1 << 128) - 1)) << 4) |
            ((lg_size & 0x7) << 132) | ((last & 1) << 135))

def unpack_write_req(val):
    return {"data": val & ((1 << 512) - 1), "vaddr": (val >> 512) & 0xFFFFFFFF, "last": (val >> 544) & 1}

def pack_write_resp(lg_size, source): return (lg_size & 0x7) | ((source & 0xF) << 3)
def unpack_rocc_resp(val): return {"rd": val & 0x1F, "data": (val >> 5) & _MASK64}

# counter event/external constants (CounterFile.scala)
class CounterEvent:
    DISABLE = 0
    MAIN_LD_CYCLES, MAIN_ST_CYCLES, MAIN_EX_CYCLES = 1, 2, 3
    MAIN_LD_ST_CYCLES, MAIN_LD_EX_CYCLES, MAIN_ST_EX_CYCLES, MAIN_LD_ST_EX_CYCLES = 4, 5, 6, 7
    LOAD_DMA_WAIT_CYCLE, LOAD_ACTIVE_CYCLE, LOAD_SCRATCHPAD_WAIT_CYCLE = 8, 9, 10
    STORE_DMA_WAIT_CYCLE, STORE_ACTIVE_CYCLE, STORE_POOLING_CYCLE, STORE_SCRATCHPAD_WAIT_CYCLE = 11, 12, 13, 14
    RDMA_ACTIVE_CYCLE, RDMA_TL_WAIT_CYCLES = 18, 20
    WDMA_ACTIVE_CYCLE, WDMA_TL_WAIT_CYCLES = 21, 23
    EXE_ACTIVE_CYCLE, EXE_FLUSH_CYCLE, EXE_CONTROL_Q_BLOCK_CYCLE = 24, 25, 26
    EXE_PRELOAD_HAZ_CYCLE, EXE_OVERLAP_HAZ_CYCLE = 27, 28
    SCRATCHPAD_A_WAIT_CYCLE, SCRATCHPAD_B_WAIT_CYCLE, SCRATCHPAD_D_WAIT_CYCLE = 29, 30, 31
    ACC_A_WAIT_CYCLE, ACC_B_WAIT_CYCLE, ACC_D_WAIT_CYCLE = 32, 33, 34
    A_GARBAGE_CYCLES, B_GARBAGE_CYCLES, D_GARBAGE_CYCLES = 35, 36, 37
    IM2COL_MEM_CYCLES, IM2COL_ACTIVE_CYCLES, IM2COL_TRANSPOSER_WAIT_CYCLE = 38, 39, 40
    RESERVATION_STATION_FULL_CYCLES, RESERVATION_STATION_ACTIVE_CYCLES = 41, 42
    LOOP_MATMUL_ACTIVE_CYCLES, TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES = 43, 44
    N = 45

class CounterExternal:
    DISABLE = 0
    RESERVATION_STATION_LD_COUNT, RESERVATION_STATION_ST_COUNT, RESERVATION_STATION_EX_COUNT = 1, 2, 3
    RDMA_BYTES_REC, WDMA_BYTES_SENT = 4, 5
    RDMA_TOTAL_LATENCY, WDMA_TOTAL_LATENCY = 6, 7
    N, EXTERNAL_WIDTH = 8, 32

# generic Decoupled stream helpers
async def stream_consumer(clk, queue, ready, valid, payload, rand=False):
    while True:
        await RisingEdge(clk)
        ready.value = random.getrandbits(1) if rand else 1
        await ReadOnly()
        if ready.value and valid.value:
            await queue.put({k: v.value for k, v in payload.items()})

async def stream_producer(clk, queue, ready, valid, payload, timeout=10000, rand=False):
    while True:
        await RisingEdge(clk)
        valid.value = 0
        if (random.getrandbits(1) if rand else 1) and queue.qsize():
            valid.value = 1
            for k, v in (await queue.get()).items(): payload[k].value = v
            await ReadWrite()
            cycles = 0
            while ready.value == 0:
                await RisingEdge(clk)
                await ReadWrite()
                cycles += 1
                assert cycles < timeout, f"stream_producer timeout ({timeout} cycles)"

# DMA interface — services 4-channel DMA packet protocol
class GemminiDMAInterface:
    def __init__(self, dut, config, mem_size=4 * 1024 * 1024, rand=False):
        self.dut, self.config, self.memory, self.rand = dut, config, np.zeros(mem_size, dtype=np.uint8), rand
        self.clock = Clock(dut.clock, 10, unit="ns")
        self.read_req_queue, self.read_resp_queue = Queue(), Queue()
        self.write_req_queue, self.write_resp_queue = Queue(), Queue()
        self._write_source = 0

    async def init(self):
        cocotb.start_soon(self.clock.start(start_high=False))
        dut = self.dut
        channels = [
            ("readReq", "consumer", ["lg_size", "bytes_read", "shift", "vaddr", "source"],
             self.read_req_queue),
            ("readResp", "producer", ["source", "data", "lg_size", "last"], self.read_resp_queue),
            ("writeReq", "consumer", ["data", "vaddr", "last"], self.write_req_queue),
            ("writeResp", "producer", ["lg_size", "source"], self.write_resp_queue),
        ]
        for name, mode, fields, queue in channels:
            pfx = f"io_{name}Packet"
            payload = {f: getattr(dut, f"{pfx}_bits_{f}") for f in fields}
            fn = stream_consumer if mode == "consumer" else stream_producer
            cocotb.start_soon(fn(dut.clock, queue, getattr(dut, f"{pfx}_ready"),
                                 getattr(dut, f"{pfx}_valid"), payload, rand=self.rand))
        cocotb.start_soon(self.memory_read_process())
        cocotb.start_soon(self.memory_write_process())
        dut.io_resp_ready.value = 1
        await self.reset()

    async def reset(self):
        self.dut.reset.value = 1
        self.dut.io_cmd_valid.value = 0
        await Timer(100, unit="ns")
        self.dut.reset.value = 0

    async def memory_read_process(self):
        beat_bytes = self.config.beat_bytes
        while True:
            req = await self.read_req_queue.get()
            lg_size, vaddr, source = int(req["lg_size"]), int(req["vaddr"]), int(req["source"])
            num_beats = max(1, (1 << lg_size) // beat_bytes)
            for i in range(num_beats):
                addr = vaddr + i * beat_bytes
                data = bytearray(beat_bytes)
                for b in range(beat_bytes):
                    if 0 <= addr + b < len(self.memory): data[b] = int(self.memory[addr + b])
                await self.read_resp_queue.put({
                    "source": source, "data": int.from_bytes(data, "little"),
                    "lg_size": lg_size, "last": int(i == num_beats - 1)})

    async def memory_write_process(self):
        beat_bytes, byte_offset, last_vaddr = self.config.beat_bytes, 0, None
        while True:
            req = await self.write_req_queue.get()
            vaddr, is_last, data_int = int(req["vaddr"]), int(req["last"]), int(req["data"])
            if vaddr != last_vaddr: byte_offset, last_vaddr = 0, vaddr
            data = (data_int & ((1 << (beat_bytes * 8)) - 1)).to_bytes(beat_bytes, "little")
            for b in range(beat_bytes):
                if 0 <= vaddr + byte_offset + b < len(self.memory):
                    self.memory[vaddr + byte_offset + b] = data[b]
            byte_offset += beat_bytes
            if is_last:
                lg = int(math.log2(byte_offset)) if byte_offset > 0 else 0
                await self.write_resp_queue.put({"lg_size": lg, "source": self._write_source})

# local address encoding
def sp_addr(addr, bits=12): return addr & ((1 << bits) - 1)
def acc_addr(addr, accumulate=False, read_full=False, bits=12):
    val = (addr & ((1 << bits) - 1)) | (1 << 31)
    if accumulate: val |= (1 << 30)
    if read_full: val |= (1 << 29)
    return val
GARBAGE_ADDR = 0xFFFFFFFF

def _log2ceil(x): return 0 if x <= 1 else (x - 1).bit_length()
def _float_to_bits(f): return struct.unpack("<I", struct.pack("<f", f))[0]

# instruction encoding
class GemminiISA:
    CONFIG, LOAD2, LOAD, STORE = 0, 1, 2, 3
    COMPUTE_AND_FLIP, COMPUTE_AND_STAY, PRELOAD, FLUSH = 4, 5, 6, 7
    COUNTER_OP = 126
    LOOP_WS = 8
    LOOP_WS_CONFIG_BOUNDS, LOOP_WS_CONFIG_ADDRS_AB, LOOP_WS_CONFIG_ADDRS_DC = 9, 10, 11
    LOOP_WS_CONFIG_STRIDES_AB, LOOP_WS_CONFIG_STRIDES_DC = 12, 13
    LOOP_CONV_WS = 15
    LOOP_CONV_WS_CONFIG_1, LOOP_CONV_WS_CONFIG_2, LOOP_CONV_WS_CONFIG_3 = 16, 17, 18
    LOOP_CONV_WS_CONFIG_4, LOOP_CONV_WS_CONFIG_5, LOOP_CONV_WS_CONFIG_6 = 19, 20, 21
    CONFIG_EX, CONFIG_LOAD, CONFIG_STORE, CONFIG_NORM = 0, 1, 2, 3
    ACT_NONE, ACT_RELU, ACT_LAYERNORM, ACT_IGELU, ACT_SOFTMAX = 0, 1, 2, 3, 4
    DATAFLOW_OS, DATAFLOW_WS = 0, 1

    @staticmethod
    def _pack_addr(addr, cols, rows):
        return (rows & 0xFFFF) << 48 | (cols & 0xFFFF) << 32 | (addr & 0xFFFFFFFF)

    @staticmethod
    def config_ex(dataflow=1, activation=0, acc_scale=0x3F800000, a_stride=1, c_stride=1,
                  in_shift=0, relu6_shift=0, set_only_strides=0, a_transpose=0, b_transpose=0):
        rs1 = ((acc_scale & 0xFFFFFFFF) << 32 | (a_stride & 0xFFFF) << 16 |
               (b_transpose & 1) << 9 | (a_transpose & 1) << 8 | (set_only_strides & 1) << 7 |
               (activation & 0x3) << 3 | (dataflow & 1) << 2 | (GemminiISA.CONFIG_EX & 0x3))
        rs2 = (c_stride & 0xFFFF) << 48 | (relu6_shift & 0xFFFF) << 32 | (in_shift & 0xFFFFFFFF)
        return (GemminiISA.CONFIG, rs1, rs2)

    @staticmethod
    def config_ld(stride, scale=0x3F800000, state_id=0, shrink=0,
                  block_mvin_stride=16, pixel_repeats=1):
        rs1 = ((scale & 0xFFFFFFFF) << 32 | (block_mvin_stride & 0xFFFF) << 16 |
               (pixel_repeats & 0xFF) << 8 | (state_id & 0x3) << 3 |
               (shrink & 1) << 2 | (GemminiISA.CONFIG_LOAD & 0x3))
        return (GemminiISA.CONFIG, rs1, stride & _MASK64)

    @staticmethod
    def config_st(stride, acc_scale=0x3F800000, activation=0, pool_stride=0, pool_size=0,
                  pool_out_dim=0, porows=0, pocols=0, orows=0, ocols=0, upad=0, lpad=0):
        rs1 = ((ocols & 0xFF) << 56 | (orows & 0xFF) << 48 | (pocols & 0xFF) << 40 |
               (porows & 0xFF) << 32 | (pool_out_dim & 0xFF) << 24 | (lpad & 0x3) << 10 |
               (upad & 0x3) << 8 | (pool_size & 0x3) << 6 | (pool_stride & 0x3) << 4 |
               (activation & 0x3) << 2 | (GemminiISA.CONFIG_STORE & 0x3))
        return (GemminiISA.CONFIG, rs1, (acc_scale & 0xFFFFFFFF) << 32 | (stride & 0xFFFFFFFF))

    @staticmethod
    def mvin(dram_addr, local_addr, cols, rows):
        return (GemminiISA.LOAD, dram_addr & _MASK64, GemminiISA._pack_addr(local_addr, cols, rows))

    @staticmethod
    def mvin2(dram_addr, local_addr, cols, rows):
        return (GemminiISA.LOAD2, dram_addr & _MASK64, GemminiISA._pack_addr(local_addr, cols, rows))

    @staticmethod
    def mvout(dram_addr, local_addr, cols, rows):
        return (GemminiISA.STORE, dram_addr & _MASK64, GemminiISA._pack_addr(local_addr, cols, rows))

    @staticmethod
    def preload(bd_addr, c_addr, bd_cols, bd_rows, c_cols, c_rows):
        return (GemminiISA.PRELOAD, GemminiISA._pack_addr(bd_addr, bd_cols, bd_rows),
                GemminiISA._pack_addr(c_addr, c_cols, c_rows))

    @staticmethod
    def compute_preloaded(a_addr, bd_addr, a_cols=0, a_rows=0, bd_cols=0, bd_rows=0):
        return (GemminiISA.COMPUTE_AND_FLIP, GemminiISA._pack_addr(a_addr, a_cols, a_rows),
                GemminiISA._pack_addr(bd_addr, bd_cols, bd_rows))

    @staticmethod
    def compute_accumulated(a_addr, bd_addr, a_cols=0, a_rows=0, bd_cols=0, bd_rows=0):
        return (GemminiISA.COMPUTE_AND_STAY, GemminiISA._pack_addr(a_addr, a_cols, a_rows),
                GemminiISA._pack_addr(bd_addr, bd_cols, bd_rows))

    @staticmethod
    def flush(skip=0): return (GemminiISA.FLUSH, skip, 0)

    # counter instructions — every command produces a response
    @staticmethod
    def counter_configure(idx, event, external=False):
        rs1 = (1 << 3) | ((idx & 0xFF) << 4) | ((event & 0x3F) << 12) | ((int(external) & 1) << 31)
        return (GemminiISA.COUNTER_OP, rs1, 0)

    @staticmethod
    def counter_reset(): return (GemminiISA.COUNTER_OP, 1, 0)
    @staticmethod
    def counter_read(idx): return (GemminiISA.COUNTER_OP, (idx & 0xFF) << 4, 0)
    @staticmethod
    def counter_snapshot(idx=0): return (GemminiISA.COUNTER_OP, (1 << 2) | ((idx & 0xFF) << 4), 0)
    @staticmethod
    def counter_snapshot_reset(idx=0): return (GemminiISA.COUNTER_OP, (1 << 1) | ((idx & 0xFF) << 4), 0)

    # loop WS
    @staticmethod
    def loop_ws_config_bounds(max_i, max_j, max_k, pad_i=0, pad_j=0, pad_k=0):
        rs1 = (pad_k & 0xFFFF) << 32 | (pad_j & 0xFFFF) << 16 | (pad_i & 0xFFFF)
        rs2 = (max_k & 0xFFFF) << 32 | (max_j & 0xFFFF) << 16 | (max_i & 0xFFFF)
        return (GemminiISA.LOOP_WS_CONFIG_BOUNDS, rs1, rs2)

    @staticmethod
    def loop_ws_config_addrs_ab(a, b):
        return (GemminiISA.LOOP_WS_CONFIG_ADDRS_AB, a & _MASK64, b & _MASK64)
    @staticmethod
    def loop_ws_config_addrs_dc(d, c):
        return (GemminiISA.LOOP_WS_CONFIG_ADDRS_DC, d & _MASK64, c & _MASK64)
    @staticmethod
    def loop_ws_config_strides_ab(a, b):
        return (GemminiISA.LOOP_WS_CONFIG_STRIDES_AB, a & _MASK64, b & _MASK64)
    @staticmethod
    def loop_ws_config_strides_dc(d, c):
        return (GemminiISA.LOOP_WS_CONFIG_STRIDES_DC, d & _MASK64, c & _MASK64)

    @staticmethod
    def loop_ws(ex_accumulate=False, full_c=False, low_d=True, activation=0,
                a_transpose=False, b_transpose=False, a_spad_id=0, b_spad_id=0, is_resadd=False):
        rs1 = ((a_spad_id & 0x3) << 18 | (b_spad_id & 0x3) << 16 | (activation & 0xFF) << 8 |
               (int(low_d) << 2) | (int(full_c) << 1) | int(ex_accumulate))
        rs2 = (int(is_resadd) << 2) | (int(b_transpose) << 1) | int(a_transpose)
        return (GemminiISA.LOOP_WS, rs1, rs2)

    # loop conv WS — returns 7 instructions
    @staticmethod
    def loop_conv_ws(batch_size, in_row_dim, in_col_dim, in_channels, out_channels,
                     out_row_dim, out_col_dim, pool_out_row_dim, pool_out_col_dim,
                     stride, padding, kernel_dim, kernel_dilation,
                     pool_size, pool_stride, pool_padding,
                     batches, porows, pocols, pochs, krows, kcols, kchs,
                     lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad,
                     orows, ocols, weights, output, bias, input_addr,
                     no_bias, no_pool, downsample, activation,
                     in_stride, weight_stride, out_stride, a_spad_id=0, b_spad_id=0):
        return [
            (GemminiISA.LOOP_CONV_WS_CONFIG_1,
             (out_channels & 0xFFFF) << 48 | (in_channels & 0xFFFF) << 32 |
             (in_row_dim & 0xFFFF) << 16 | (batch_size & 0xFFFF),
             (padding & 0xFF) << 56 | (stride & 0xFF) << 48 |
             (out_col_dim & 0xFFFF) << 32 | (pool_out_row_dim & 0xFFFF) << 16 |
             (out_row_dim & 0xFFFF)),
            (GemminiISA.LOOP_CONV_WS_CONFIG_2,
             (kernel_dim & 0xFFFF) << 48 | (pool_out_col_dim & 0xFFFF) << 32 |
             (pool_size & 0xFFFF) << 16 | (pool_stride & 0xFF) << 8 | (pool_padding & 0xFF),
             (batches & 0xFFFF) << 48 | (porows & 0xFFFF) << 32 |
             (pocols & 0xFFFF) << 16 | (pochs & 0xFFFF)),
            (GemminiISA.LOOP_CONV_WS_CONFIG_3,
             (krows & 0xFFFF) << 48 | (kcols & 0xFFFF) << 32 |
             (kchs & 0xFFFF) << 16 | (lpad & 0xFFFF),
             (rpad & 0xFFFF) << 48 | (upad & 0xFFFF) << 32 |
             (dpad & 0xFF) << 24 | (plpad & 0xFF) << 16 | (in_col_dim & 0xFFFF)),
            (GemminiISA.LOOP_CONV_WS_CONFIG_4,
             (orows & 0xFFFF) << 48 | (prpad & 0xFFFF) << 32 |
             (pupad & 0x7FF) << 21 | (pdpad & 0x7FF) << 10 | (kernel_dilation & 0x3FF),
             (in_stride & 0xFFFF) << 48 | (weight_stride & 0xFFFF) << 32 |
             (out_stride & 0xFFFF) << 16 | (ocols & 0xFFFF)),
            (GemminiISA.LOOP_CONV_WS_CONFIG_5, weights & _MASK64, output & _MASK64),
            (GemminiISA.LOOP_CONV_WS_CONFIG_6, bias & _MASK64, input_addr & _MASK64),
            (GemminiISA.LOOP_CONV_WS,
             (a_spad_id & 0x3) << 18 | (b_spad_id & 0x3) << 16 | (1 << 8) | int(no_bias),
             (activation & 0x3) << 3 | int(downsample) << 1 | int(no_pool)),
        ]

# raw Gemmini command driver (io_cmd Decoupled — flat Chisel signal names)
_MSTATUS = {
    "debug": 0, "cease": 0, "wfi": 0, "isa": 0, "dprv": 3, "dv": 0, "prv": 3, "v": 0,
    "sd": 0, "zero2": 0, "mpv": 0, "gva": 0, "mbe": 0, "sbe": 0, "sxl": 0, "uxl": 0,
    "sd_rv32": 0, "zero1": 0, "tsr": 0, "tw": 0, "tvm": 0, "mxr": 0, "sum": 0, "mprv": 0,
    "xs": 0, "fs": 0, "mpp": 3, "vs": 0, "spp": 0, "mpie": 0, "ube": 0, "spie": 0,
    "upie": 0, "mie": 0, "hie": 0, "sie": 0, "uie": 0,
}

async def send_raw_cmd(dut, funct, rs1, rs2):
    await RisingEdge(dut.clock)
    dut.io_cmd_bits_inst_funct.value = funct & 0x7F
    dut.io_cmd_bits_rs1.value = rs1 & _MASK64
    dut.io_cmd_bits_rs2.value = rs2 & _MASK64
    for k, v in {"rs1": 0, "rs2": 0, "xd": 0, "xs1": 1, "xs2": 1, "rd": 0, "opcode": 0x0B}.items():
        getattr(dut, f"io_cmd_bits_inst_{k}").value = v
    for k, v in _MSTATUS.items():
        getattr(dut, f"io_cmd_bits_status_{k}").value = v
    dut.io_cmd_valid.value = 1
    await ReadWrite()
    cycles = 0
    while dut.io_cmd_ready.value == 0:
        await RisingEdge(dut.clock)
        await ReadWrite()
        cycles += 1
        assert cycles < 10000, "timeout waiting for io_cmd_ready"
    await RisingEdge(dut.clock)
    dut.io_cmd_valid.value = 0

async def wait_raw_not_busy(dut, timeout=100000):
    for _ in range(timeout):
        await RisingEdge(dut.clock)
        await ReadOnly()
        if dut.io_busy.value == 0: return
    raise TimeoutError(f"Gemmini still busy after {timeout} cycles")

# asm parser
_DATAFLOW_MAP = {"os": 0, "OS": 0, "ws": 1, "WS": 1}
_ACTIVATION_MAP = {"none": 0, "NONE": 0, "relu": 1, "RELU": 1, "layernorm": 2, "LAYERNORM": 2,
                   "igelu": 3, "IGELU": 3, "softmax": 4, "SOFTMAX": 4}

def _parse_addr(s, bits=12):
    s = s.strip()
    if s == "garbage": return GARBAGE_ADDR
    if s.startswith("sp:"): return sp_addr(int(s[3:]), bits)
    if s.startswith("acc+:"): return acc_addr(int(s[5:]), accumulate=True, bits=bits)
    if s.startswith("accf:"): return acc_addr(int(s[5:]), read_full=True, bits=bits)
    if s.startswith("acc:"): return acc_addr(int(s[4:]), bits=bits)
    return int(s, 16) if s.startswith(("0x", "0X")) else int(s)

def _parse_scale(s):
    s = s.strip()
    if s.startswith(("0x", "0X")): return int(s, 16)
    try: return _float_to_bits(float(s))
    except ValueError: return int(s)

def _parse_kwargs(tokens):
    kwargs, positional = {}, []
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kwargs[k] = v
        else: positional.append(t)
    return positional, kwargs

def _resolve_named(val, m): return m.get(val, int(val))

def parse_gemmini_asm(text, addr_bits=12):
    instrs = []
    for line in text.strip().splitlines():
        line = line.split("#")[0].strip()
        if not line: continue
        tokens = line.split()
        cmd, args = tokens[0].lower(), tokens[1:]
        pos, kw = _parse_kwargs(args)

        if cmd == "config_ex":
            d = {}
            if "dataflow" in kw: d["dataflow"] = _resolve_named(kw["dataflow"], _DATAFLOW_MAP)
            if "activation" in kw: d["activation"] = _resolve_named(kw["activation"], _ACTIVATION_MAP)
            if "acc_scale" in kw: d["acc_scale"] = _parse_scale(kw["acc_scale"])
            for k in ("a_stride", "c_stride", "in_shift", "relu6_shift", "a_transpose", "b_transpose"):
                if k in kw: d[k] = int(kw[k])
            instrs.append(GemminiISA.config_ex(**d))

        elif cmd == "config_ld":
            d = {"stride": int(kw.get("stride", pos[0] if pos else "0"))}
            if "scale" in kw: d["scale"] = _parse_scale(kw["scale"])
            for k in ("state_id", "shrink", "block_mvin_stride", "pixel_repeats"):
                if k in kw: d[k] = int(kw[k])
            instrs.append(GemminiISA.config_ld(**d))

        elif cmd == "config_st":
            d = {"stride": int(kw.get("stride", pos[0] if pos else "0"))}
            if "acc_scale" in kw: d["acc_scale"] = _parse_scale(kw["acc_scale"])
            if "activation" in kw: d["activation"] = _resolve_named(kw["activation"], _ACTIVATION_MAP)
            for k in ("pool_stride", "pool_size"):
                if k in kw: d[k] = int(kw[k])
            instrs.append(GemminiISA.config_st(**d))

        elif cmd in ("mvin", "mvin2"):
            dram = _parse_addr(pos[0], addr_bits)
            la = _parse_addr(pos[1], addr_bits)
            cols, rows = int(kw.get("cols", "16")), int(kw.get("rows", "16"))
            fn = GemminiISA.mvin if cmd == "mvin" else GemminiISA.mvin2
            instrs.append(fn(dram, la, cols, rows))

        elif cmd == "mvout":
            dram = _parse_addr(pos[0], addr_bits)
            la = _parse_addr(pos[1], addr_bits)
            instrs.append(GemminiISA.mvout(dram, la, int(kw.get("cols", "16")),
                                           int(kw.get("rows", "16"))))

        elif cmd == "preload":
            bd = _parse_addr(pos[0], addr_bits)
            c = _parse_addr(pos[1], addr_bits)
            dc, dr = kw.get("cols", "16"), kw.get("rows", "16")
            instrs.append(GemminiISA.preload(bd, c, int(kw.get("bd_cols", dc)),
                          int(kw.get("bd_rows", dr)), int(kw.get("c_cols", dc)),
                          int(kw.get("c_rows", dr))))

        elif cmd in ("compute_preloaded", "compute_accumulated"):
            a = _parse_addr(pos[0], addr_bits)
            bd = _parse_addr(pos[1], addr_bits)
            d = {k: int(kw.get(k, "0")) for k in ("a_cols", "a_rows", "bd_cols", "bd_rows")}
            fn = GemminiISA.compute_preloaded if "pre" in cmd else GemminiISA.compute_accumulated
            instrs.append(fn(a, bd, **d))

        elif cmd == "flush":
            instrs.append(GemminiISA.flush(int(kw.get("skip", pos[0] if pos else "0"))))
        else: raise ValueError(f"Unknown Gemmini asm instruction: {cmd!r}")

    return instrs

# high-level runner (ties DMA + commands together)
class GemminiRunner:
    def __init__(self, dut, config, mem_size=4 * 1024 * 1024, rand=False):
        self.dut, self.config = dut, config
        self.dma = GemminiDMAInterface(dut, config, mem_size, rand=rand)
        self.addr_bits = _log2ceil(max(config.sp_rows, config.acc_rows))

    async def init(self): await self.dma.init()

    async def execute(self, instrs):
        if isinstance(instrs, str): instrs = parse_gemmini_asm(instrs, addr_bits=self.addr_bits)
        for f, rs1, rs2 in instrs: await send_raw_cmd(self.dut, f, rs1, rs2)

    async def wait_not_busy(self, timeout=100000): await wait_raw_not_busy(self.dut, timeout)

    def load_matrix(self, addr, matrix, dtype=np.int8):
        flat = np.ascontiguousarray(matrix, dtype=dtype).view(np.uint8).ravel()
        assert addr + len(flat) <= len(self.dma.memory), f"matrix at 0x{addr:x} exceeds memory"
        self.dma.memory[addr:addr + len(flat)] = flat

    def read_matrix(self, addr, rows, cols, dtype=np.int8):
        nbytes = rows * cols * np.dtype(dtype).itemsize
        return np.frombuffer(self.dma.memory[addr:addr + nbytes].copy(), dtype=dtype).reshape(rows, cols)

# instruction stream generator
class GemminiProgram:
    MVIN_SCALE_IDENTITY = ACC_SCALE_IDENTITY = 0x3F800000

    def __init__(self, config):
        self.config = config

    def _ceildiv(self, a, b): return (a + b - 1) // b

    def matmul_ws(self, M, N, K, A_addr, B_addr, C_addr, D_addr=None, A_stride=None,
                  B_stride=None, C_stride=None, D_stride=None, activation=GemminiISA.ACT_NONE,
                  A_scale=None, B_scale=None, D_scale=None, acc_scale=None,
                  full_c=False, low_d=True, a_transpose=False, b_transpose=False,
                  repeating_bias=False):
        dim, ib, ab = self.config.block_rows, self.config.input_bytes, self.config.acc_bytes
        if A_stride is None: A_stride = K
        if B_stride is None: B_stride = N
        if C_stride is None: C_stride = N
        if D_stride is None: D_stride = 0 if repeating_bias else N
        if A_scale is None: A_scale = self.MVIN_SCALE_IDENTITY
        if B_scale is None: B_scale = self.MVIN_SCALE_IDENTITY
        if D_scale is None: D_scale = self.MVIN_SCALE_IDENTITY
        if acc_scale is None: acc_scale = self.ACC_SCALE_IDENTITY
        sizeof_c = ab if full_c else ib
        sizeof_d = ib if low_d else ab

        tile_i, tile_j, tile_k = self._compute_tile_factors(M, N, K)
        I_pad = self._ceildiv(M, dim) * dim
        J_pad = self._ceildiv(N, dim) * dim
        K_pad = self._ceildiv(K, dim) * dim
        pad_I, pad_J, pad_K = I_pad - M, J_pad - N, K_pad - K
        I0 = self._ceildiv(I_pad, tile_i * dim)
        J0 = self._ceildiv(J_pad, tile_j * dim)
        K0 = self._ceildiv(K_pad, tile_k * dim)
        last_I = tile_i if I_pad % (tile_i * dim) == 0 else (I_pad // dim) % tile_i
        last_J = tile_j if J_pad % (tile_j * dim) == 0 else (J_pad // dim) % tile_j
        last_K = tile_k if K_pad % (tile_k * dim) == 0 else (K_pad // dim) % tile_k

        instrs = [
            GemminiISA.config_ex(dataflow=GemminiISA.DATAFLOW_WS, activation=activation,
                                 acc_scale=acc_scale, a_transpose=int(a_transpose),
                                 b_transpose=int(b_transpose)),
            GemminiISA.config_st(stride=C_stride * sizeof_c, activation=activation,
                                 acc_scale=acc_scale),
            GemminiISA.config_ld(stride=A_stride * ib, scale=A_scale, state_id=0),
            GemminiISA.config_ld(stride=B_stride * ib, scale=B_scale, state_id=1),
            GemminiISA.config_ld(stride=D_stride * sizeof_d if not repeating_bias else 0,
                                 scale=D_scale, shrink=int(low_d), state_id=2),
        ]

        for i0 in range(I0):
            for j0 in range(J0):
                for k0 in range(K0):
                    I = last_I if i0 == I0 - 1 else tile_i
                    J = last_J if j0 == J0 - 1 else tile_j
                    K_t = last_K if k0 == K0 - 1 else tile_k
                    pi = pad_I if i0 == I0 - 1 else 0
                    pj = pad_J if j0 == J0 - 1 else 0
                    pk = pad_K if k0 == K0 - 1 else 0

                    if a_transpose:
                        a = A_addr + (k0 * tile_k * dim * A_stride + i0 * tile_i * dim) * ib
                    else:
                        a = A_addr + (i0 * tile_i * dim * A_stride + k0 * tile_k * dim) * ib
                    if b_transpose:
                        b = B_addr + (j0 * tile_j * dim * B_stride + k0 * tile_k * dim) * ib
                    else:
                        b = B_addr + (k0 * tile_k * dim * B_stride + j0 * tile_j * dim) * ib

                    if D_addr is not None and k0 == 0:
                        br = 0 if repeating_bias else i0 * tile_i * dim
                        d = D_addr + (br * D_stride + j0 * tile_j * dim) * sizeof_d
                    else: d = 0

                    c = (C_addr + (i0 * tile_i * dim * C_stride +
                         j0 * tile_j * dim) * sizeof_c if k0 == K0 - 1 else 0)

                    instrs.extend([
                        GemminiISA.loop_ws_config_bounds(I, J, K_t, pi, pj, pk),
                        GemminiISA.loop_ws_config_addrs_ab(a, b),
                        GemminiISA.loop_ws_config_addrs_dc(d, c),
                        GemminiISA.loop_ws_config_strides_ab(A_stride, B_stride),
                        GemminiISA.loop_ws_config_strides_dc(D_stride, C_stride),
                        GemminiISA.loop_ws(ex_accumulate=(k0 > 0), full_c=full_c, low_d=low_d,
                                           activation=activation, a_transpose=a_transpose,
                                           b_transpose=b_transpose),
                    ])

        return instrs

    def conv2d_ws(self, batch_size, in_channels, out_channels, in_row_dim, in_col_dim,
                  kernel_dim, stride, padding, input_addr, weights_addr, output_addr,
                  bias_addr=None, activation=GemminiISA.ACT_NONE, acc_scale=None,
                  pool_size=0, pool_stride=0, pool_padding=0):
        ib, ab = self.config.input_bytes, self.config.acc_bytes
        if acc_scale is None: acc_scale = self.ACC_SCALE_IDENTITY
        no_bias = bias_addr is None
        if no_bias: bias_addr = 1
        no_pool = pool_stride == 0
        if no_pool: pool_size, pool_stride, pool_padding = 1, 1, 0

        out_row_dim = (in_row_dim + 2 * padding - kernel_dim) // stride + 1
        out_col_dim = (in_col_dim + 2 * padding - kernel_dim) // stride + 1
        downsample = (stride == 2 and kernel_dim == 1 and padding == 0 and no_pool
                      and in_row_dim % 2 == 0 and in_col_dim % 2 == 0)
        pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) // pool_stride + 1
        pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) // pool_stride + 1
        in_stride, weight_stride, out_stride = in_channels, out_channels, out_channels

        batches, orows, ocols, ochs, krows, kcols, kchs = self._compute_conv_tile_factors(
            batch_size, out_row_dim, out_col_dim, out_channels, kernel_dim, kernel_dim,
            in_channels, stride, downsample, pool_size, pool_stride)
        porows = (orows - pool_size + pool_stride) // pool_stride if not no_pool else orows
        pocols = (ocols - pool_size + pool_stride) // pool_stride if not no_pool else ocols
        pochs = ochs

        eff_stride = stride >> int(downsample)
        instrs = [
            GemminiISA.config_st(stride=out_stride * ib, activation=activation,
                                 acc_scale=acc_scale),
            GemminiISA.config_ex(dataflow=GemminiISA.DATAFLOW_WS, a_stride=eff_stride),
            GemminiISA.config_ld(stride=in_stride * ib * (2 if downsample else 1), state_id=0),
            GemminiISA.config_ld(stride=weight_stride * ib, state_id=1),
            GemminiISA.config_ld(stride=0, state_id=2),
        ]

        for b_idx in range(0, batch_size, batches):
            for porow in range(0, pool_out_row_dim, porows):
                orow = porow * pool_stride - pool_padding
                for pocol in range(0, pool_out_col_dim, pocols):
                    ocol = pocol * pool_stride - pool_padding
                    for poch in range(0, out_channels, pochs):
                        for kr in range(0, kernel_dim, krows):
                            orow_f = max(0, orow)
                            irow = orow_f * stride + kr - padding
                            for kc in range(0, kernel_dim, kcols):
                                ocol_f = max(0, ocol)
                                icol = ocol_f * stride + kc - padding
                                for kch in range(0, in_channels, kchs):
                                    b_ = min(batches, batch_size - b_idx)
                                    pr_ = min(porows, pool_out_row_dim - porow)
                                    pc_ = min(pocols, pool_out_col_dim - pocol)
                                    pch_ = min(pochs, out_channels - poch)
                                    kr_ = min(krows, kernel_dim - kr)
                                    kc_ = min(kcols, kernel_dim - kc)
                                    kch_ = min(kchs, in_channels - kch)
                                    or_ = pr_ * pool_stride + pool_size - 1
                                    oc_ = pc_ * pool_stride + pool_size - 1

                                    plpad = max(0, -ocol)
                                    prpad = max(0, ocol + oc_ - out_col_dim)
                                    pupad = max(0, -orow)
                                    pdpad = max(0, orow + or_ - out_row_dim)
                                    icols_ = (oc_ - plpad - prpad) * stride + kc_ - 1
                                    irows_ = (or_ - pupad - pdpad) * stride + kr_ - 1
                                    lpad_ = max(0, -icol)
                                    rpad_ = max(0, icol + icols_ - in_col_dim)
                                    upad_ = max(0, -irow)
                                    dpad_ = max(0, irow + irows_ - in_row_dim)

                                    last_k = (kr + kr_ >= kernel_dim
                                              and kc + kc_ >= kernel_dim
                                              and kch + kch_ >= in_channels)
                                    o_addr = (output_addr +
                                              (b_idx * pool_out_row_dim * pool_out_col_dim +
                                               porow * pool_out_col_dim + pocol) *
                                              out_stride * ib + poch * ib) if last_k else 0
                                    bi_ = (bias_addr + poch * ab
                                           if kr == 0 and kc == 0
                                           and kch == 0 and not no_bias else 0)
                                    in_a = (input_addr +
                                            (b_idx * in_row_dim * in_col_dim +
                                             max(0, irow + upad_) * in_col_dim +
                                             max(0, icol + lpad_)) *
                                            in_stride * ib + kch * ib)
                                    w_a = (weights_addr +
                                           (kr * kernel_dim * in_channels +
                                            kc * in_channels + kch) *
                                           weight_stride * ib + poch * ib)

                                    instrs.extend(GemminiISA.loop_conv_ws(
                                        batch_size=batch_size,
                                        in_row_dim=in_row_dim, in_col_dim=in_col_dim,
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        out_row_dim=out_row_dim,
                                        out_col_dim=out_col_dim,
                                        pool_out_row_dim=pool_out_row_dim,
                                        pool_out_col_dim=pool_out_col_dim,
                                        stride=stride, padding=padding,
                                        kernel_dim=kernel_dim, kernel_dilation=1,
                                        pool_size=pool_size,
                                        pool_stride=pool_stride,
                                        pool_padding=pool_padding,
                                        batches=b_, porows=pr_, pocols=pc_,
                                        pochs=pch_, krows=kr_, kcols=kc_,
                                        kchs=kch_,
                                        lpad=lpad_, rpad=rpad_, upad=upad_,
                                        dpad=dpad_, plpad=plpad, prpad=prpad,
                                        pupad=pupad, pdpad=pdpad,
                                        orows=or_, ocols=oc_,
                                        weights=w_a, output=o_addr,
                                        bias=bi_, input_addr=in_a,
                                        no_bias=(bi_ == 0), no_pool=no_pool,
                                        downsample=downsample,
                                        activation=activation,
                                        in_stride=in_stride,
                                        weight_stride=weight_stride,
                                        out_stride=out_stride))

        return instrs

    def _compute_conv_tile_factors(self, batch_size, out_row_dim, out_col_dim, out_channels,
                                   kernel_row, kernel_col, in_channels, stride, downsample,
                                   pool_size, pool_stride):
        dim, max_sp, max_acc = self.config.block_rows, self.config.sp_rows // 2, self.config.acc_rows // 2
        def _spad(b, or_, oc_, och, kr, kc, kch):
            ir = (or_ + kr - 1) * stride if not downsample else or_
            ic = (oc_ + kc - 1) * stride if not downsample else oc_
            return (self._ceildiv(kch, dim) * b * ir * ic +
                    self._ceildiv(och, dim) * kr * kc * kch)
        def _acc(b, or_, oc_, och): return self._ceildiv(och, dim) * b * or_ * oc_
        def _fits(a): return _spad(*a) <= max_sp and _acc(*a[:4]) <= max_acc

        args = [batch_size, 1, 1, min(out_channels, dim),
                kernel_row, kernel_col, min(in_channels, dim)]
        maxs = [batch_size, out_row_dim, out_col_dim,
                self._ceildiv(out_channels, dim) * dim,
                kernel_row, kernel_col, self._ceildiv(in_channels, dim) * dim]

        if not _fits(args):
            while not _fits(args) and args[0] > 1: args[0] -= 1

        for idx in [2, 1, 3, 6, 0]:
            while args[idx] < maxs[idx]:
                step = dim if idx in (3, 6) else 1
                candidate = list(args)
                candidate[idx] = min(args[idx] + step, maxs[idx])
                if _fits(candidate): args = candidate
                else: break

        return tuple(args)

    def _compute_tile_factors(self, M, N, K):
        dim, sp_rows, acc_rows = self.config.block_rows, self.config.sp_rows, self.config.acc_rows
        ti, tj, tk = self._ceildiv(M, dim), self._ceildiv(N, dim), self._ceildiv(K, dim)
        while True:
            if (ti * tk + tj * tk) * dim <= sp_rows and ti * tj * dim <= acc_rows: break
            if tk >= ti and tk >= tj and tk > 1: tk -= 1
            elif ti >= tj and ti > 1: ti -= 1
            elif tj > 1: tj -= 1
            else: break
        return (ti, tj, tk)

    def print_program(self, instrs):
        FUNCT_NAMES = {
            0: "CONFIG", 1: "LOAD2", 2: "LOAD", 3: "STORE", 4: "COMPUTE_AND_FLIP",
            5: "COMPUTE_AND_STAY", 6: "PRELOAD", 7: "FLUSH", 8: "LOOP_WS",
            9: "LOOP_WS_BOUNDS", 10: "LOOP_WS_ADDRS_AB", 11: "LOOP_WS_ADDRS_DC",
            12: "LOOP_WS_STRIDES_AB", 13: "LOOP_WS_STRIDES_DC", 15: "LOOP_CONV_WS",
            16: "CONV_CFG_1", 17: "CONV_CFG_2", 18: "CONV_CFG_3", 19: "CONV_CFG_4",
            20: "CONV_CFG_5", 21: "CONV_CFG_6", 126: "COUNTER_OP",
        }
        for i, (funct, rs1, rs2) in enumerate(instrs):
            name = FUNCT_NAMES.get(funct, f"FUNCT_{funct}")
            print(f"  [{i:3d}] {name:25s}  rs1=0x{rs1:016x}  rs2=0x{rs2:016x}")

if __name__ == "__main__":
    import argparse, json as _json
    from types import SimpleNamespace

    def _cli_config(dim, input_bytes, acc_bytes, sp_capacity_kb=256, acc_capacity_kb=64,
                    sp_banks=4, acc_banks=2, dma_buswidth=128):
        """Build a lightweight config namespace for CLI use (no Verilog generation)."""
        input_width = input_bytes * 8
        acc_width = acc_bytes * 8
        block_rows = block_cols = dim
        return SimpleNamespace(
            block_rows=block_rows, block_cols=block_cols,
            input_bytes=input_bytes, acc_bytes=acc_bytes,
            input_width=input_width, acc_width=acc_width,
            sp_banks=sp_banks, acc_banks=acc_banks,
            sp_bank_entries=sp_capacity_kb * 1024 * 8 // (sp_banks * block_cols * input_width),
            acc_bank_entries=acc_capacity_kb * 1024 * 8 // (acc_banks * block_cols * acc_width),
            sp_rows=sp_banks * (sp_capacity_kb * 1024 * 8 // (sp_banks * block_cols * input_width)),
            acc_rows=acc_banks * (acc_capacity_kb * 1024 * 8 // (acc_banks * block_cols * acc_width)),
            beat_bytes=dma_buswidth // 8,
        )

    parser = argparse.ArgumentParser(description="Gemmini instruction generator")
    sub = parser.add_subparsers(dest="command")

    mm = sub.add_parser("matmul")
    mm.add_argument("M", type=int)
    mm.add_argument("N", type=int)
    mm.add_argument("K", type=int)
    mm.add_argument("--a-addr", type=lambda x: int(x, 0), default=0x0)
    mm.add_argument("--b-addr", type=lambda x: int(x, 0), default=None)
    mm.add_argument("--c-addr", type=lambda x: int(x, 0), default=None)
    mm.add_argument("--d-addr", type=lambda x: int(x, 0), default=None)
    mm.add_argument("--dim", type=int, default=16)
    mm.add_argument("--input-bytes", type=int, default=1)
    mm.add_argument("--acc-bytes", type=int, default=4)
    mm.add_argument("--json", action="store_true")

    cv = sub.add_parser("conv")
    cv.add_argument("--batch", type=int, default=1)
    cv.add_argument("--in-c", type=int, required=True)
    cv.add_argument("--out-c", type=int, required=True)
    cv.add_argument("--in-h", type=int, required=True)
    cv.add_argument("--in-w", type=int, default=None)
    cv.add_argument("--kernel", type=int, required=True)
    cv.add_argument("--stride", type=int, default=1)
    cv.add_argument("--padding", type=int, default=0)
    cv.add_argument("--activation", type=int, default=0)
    cv.add_argument("--pool-size", type=int, default=0)
    cv.add_argument("--pool-stride", type=int, default=0)
    cv.add_argument("--pool-padding", type=int, default=0)
    cv.add_argument("--input-addr", type=lambda x: int(x, 0), default=0x0)
    cv.add_argument("--weights-addr", type=lambda x: int(x, 0), default=None)
    cv.add_argument("--output-addr", type=lambda x: int(x, 0), default=None)
    cv.add_argument("--bias-addr", type=lambda x: int(x, 0), default=None)
    cv.add_argument("--dim", type=int, default=16)
    cv.add_argument("--input-bytes", type=int, default=1)
    cv.add_argument("--acc-bytes", type=int, default=4)
    cv.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if hasattr(args, 'in_w') and args.in_w is None and hasattr(args, 'in_h'):
        args.in_w = args.in_h

    if args.command == "matmul":
        ib = args.input_bytes
        a_addr = args.a_addr
        b_addr = args.b_addr if args.b_addr is not None else a_addr + args.M * args.K * ib
        c_addr = args.c_addr if args.c_addr is not None else b_addr + args.K * args.N * ib
        cfg = _cli_config(args.dim, ib, args.acc_bytes)
        prog = GemminiProgram(cfg)
        instrs = prog.matmul_ws(args.M, args.N, args.K, a_addr, b_addr, c_addr,
                                D_addr=args.d_addr)
        if args.json:
            print(_json.dumps([{"funct": f, "rs1": r1, "rs2": r2}
                               for f, r1, r2 in instrs], indent=2))
        else:
            print(f"Gemmini WS matmul: C[{args.M}x{args.N}] = "
                  f"A[{args.M}x{args.K}] @ B[{args.K}x{args.N}]")
            print(f"  A @ 0x{a_addr:x}, B @ 0x{b_addr:x}, C @ 0x{c_addr:x}")
            print(f"  {len(instrs)} instructions:")
            prog.print_program(instrs)

    elif args.command == "conv":
        ib = args.input_bytes
        cfg = _cli_config(args.dim, ib, args.acc_bytes)
        prog = GemminiProgram(cfg)
        in_size = args.batch * args.in_h * args.in_w * args.in_c * ib
        w_size = args.kernel ** 2 * args.in_c * args.out_c * ib
        input_addr = args.input_addr
        weights_addr = (args.weights_addr if args.weights_addr is not None
                        else input_addr + in_size)
        bias_addr = args.bias_addr
        out_h = (args.in_h + 2 * args.padding - args.kernel) // args.stride + 1
        out_w = (args.in_w + 2 * args.padding - args.kernel) // args.stride + 1
        output_addr = (args.output_addr if args.output_addr is not None
                       else weights_addr + w_size)
        instrs = prog.conv2d_ws(
            batch_size=args.batch, in_channels=args.in_c, out_channels=args.out_c,
            in_row_dim=args.in_h, in_col_dim=args.in_w, kernel_dim=args.kernel,
            stride=args.stride, padding=args.padding, input_addr=input_addr,
            weights_addr=weights_addr, output_addr=output_addr, bias_addr=bias_addr,
            activation=args.activation, pool_size=args.pool_size,
            pool_stride=args.pool_stride, pool_padding=args.pool_padding)
        if args.json:
            print(_json.dumps([{"funct": f, "rs1": r1, "rs2": r2}
                               for f, r1, r2 in instrs], indent=2))
        else:
            print(f"Gemmini WS conv2d: [{args.batch},{args.in_c},{args.in_h},{args.in_w}] "
                  f"* [{args.out_c},{args.in_c},{args.kernel},{args.kernel}] "
                  f"stride={args.stride} pad={args.padding}")
            print(f"  output: [{args.batch},{args.out_c},{out_h},{out_w}]")
            print(f"  {len(instrs)} instructions:")
            prog.print_program(instrs)

    else: parser.print_help()
