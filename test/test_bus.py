"""Unit tests for the read/write bridge FSMs from Gemmini.
"""

import amaranth as am
from amaranth.lib.wiring import In, Out, Component
from amaranth.utils import exact_log2

import cocotb
from cocotb.triggers import RisingEdge, ReadWrite

from gemmini_amaranth.bus import AXI4, BurstType
from test.helpers import cocotb_run, cocotb_init, config_from_component


class ReadBridge(Component):
    """readReqPacket → AXI4 AR/R → readRespPacket.  ratio=1 only."""

    def __init__(self, *, vaddr_width=32, source_width=4,
                 dma_buswidth=128):
        self.vaddr_width = vaddr_width
        self.source_width = source_width
        self.dma_buswidth = dma_buswidth
        self.dma_beat_bytes = dma_buswidth // 8
        self.axi_beat_bytes = dma_buswidth // 8

        super().__init__({
            # DMA request (from Gemmini)
            "req_valid":    In(1),
            "req_ready":    Out(1),
            "req_lg_size":  In(3),
            "req_vaddr":    In(vaddr_width),
            "req_source":   In(source_width),
            # DMA response (to Gemmini)
            "resp_valid":   Out(1),
            "resp_ready":   In(1),
            "resp_source":  Out(source_width),
            "resp_data":    Out(dma_buswidth),
            "resp_lg_size": Out(3),
            "resp_last":    Out(1),
            # AXI4 read channels
            "bus": Out(AXI4(addr_width=vaddr_width, data_width=dma_buswidth)),
        })

    def elaborate(self, _):
        m = am.Module()
        dma_beat_bytes = self.dma_beat_bytes
        axi_beat_bytes = self.axi_beat_bytes

        rd_source_r = am.Signal(self.source_width)
        rd_lg_size_r = am.Signal(3)
        rd_vaddr_r = am.Signal(self.vaddr_width)
        rd_arlen_r = am.Signal(8)

        m.d.comb += [
            self.bus.arburst.eq(BurstType.INCR),
            self.bus.arsize.eq(exact_log2(axi_beat_bytes)),
        ]

        with m.FSM(name="read_bridge"):
            with m.State("IDLE"):
                m.d.comb += self.req_ready.eq(1)
                with m.If(self.req_valid):
                    m.d.sync += [
                        rd_source_r.eq(self.req_source),
                        rd_lg_size_r.eq(self.req_lg_size),
                        rd_vaddr_r.eq(self.req_vaddr),
                    ]
                    for lg in range(8):
                        with m.If(self.req_lg_size == lg):
                            dma_beats = max(1, (1 << lg) // dma_beat_bytes)
                            axi_beats = dma_beats  # ratio=1
                            m.d.sync += rd_arlen_r.eq(axi_beats - 1)
                    m.next = "SEND_AR"

            with m.State("SEND_AR"):
                m.d.comb += [
                    self.bus.araddr.eq(rd_vaddr_r),
                    self.bus.arlen.eq(rd_arlen_r),
                    self.bus.arvalid.eq(1),
                ]
                with m.If(self.bus.arready):
                    m.next = "RECV_R"

            with m.State("RECV_R"):
                m.d.comb += [
                    self.resp_valid.eq(self.bus.rvalid),
                    self.resp_data.eq(self.bus.rdata),
                    self.resp_source.eq(rd_source_r),
                    self.resp_lg_size.eq(rd_lg_size_r),
                    self.resp_last.eq(self.bus.rlast),
                    self.bus.rready.eq(self.resp_ready),
                ]
                with m.If(self.bus.rvalid & self.bus.rready & self.bus.rlast):
                    m.next = "DONE"

            with m.State("DONE"):
                m.next = "IDLE"

        return m


class WriteBridge(Component):
    """writeReqPacket → AXI4 AW/W/B → writeRespPacket.  ratio=1 only."""

    def __init__(self, *, vaddr_width=32, source_width=4,
                 dma_buswidth=128, write_data_width=512):
        self.vaddr_width = vaddr_width
        self.source_width = source_width
        self.dma_buswidth = dma_buswidth
        self.write_data_width = write_data_width
        self.dma_beat_bytes = dma_buswidth // 8
        self.axi_beat_bytes = dma_buswidth // 8
        self.beats_per_row = write_data_width // dma_buswidth

        super().__init__({
            # DMA write request (from Gemmini)
            "req_valid":    In(1),
            "req_ready":    Out(1),
            "req_data":     In(write_data_width),
            "req_vaddr":    In(vaddr_width),
            "req_last":     In(1),
            # DMA write response (to Gemmini)
            "resp_valid":   Out(1),
            "resp_ready":   In(1),
            "resp_lg_size": Out(3),
            "resp_source":  Out(source_width),
            # AXI4 write channels
            "bus": Out(AXI4(addr_width=vaddr_width, data_width=dma_buswidth)),
        })

    def elaborate(self, _):
        m = am.Module()
        axi_beat_bytes = self.axi_beat_bytes
        dma_beat_bytes = self.dma_beat_bytes
        beats_per_row = self.beats_per_row
        axi_beats_per_row = beats_per_row  # ratio=1

        wr_vaddr_r = am.Signal(self.vaddr_width)
        wr_source_counter = am.Signal(self.source_width)
        wr_dma_data = self.req_data[:self.dma_buswidth]

        m.d.comb += [
            self.bus.awburst.eq(BurstType.INCR),
            self.bus.awsize.eq(exact_log2(axi_beat_bytes)),
        ]

        with m.FSM(name="write_bridge"):
            with m.State("IDLE"):
                m.d.comb += self.req_ready.eq(0)
                with m.If(self.req_valid):
                    m.d.sync += wr_vaddr_r.eq(self.req_vaddr)
                    m.next = "SEND_AW"

            with m.State("SEND_AW"):
                m.d.comb += [
                    self.bus.awaddr.eq(wr_vaddr_r),
                    self.bus.awlen.eq(axi_beats_per_row - 1),
                    self.bus.awvalid.eq(1),
                    self.req_ready.eq(0),
                ]
                with m.If(self.bus.awready):
                    m.next = "SEND_W"

            with m.State("SEND_W"):
                m.d.comb += [
                    self.bus.wdata.eq(wr_dma_data),
                    self.bus.wvalid.eq(self.req_valid),
                    self.bus.wlast.eq(self.req_last),
                    self.req_ready.eq(self.bus.wready),
                ]
                with m.If(self.bus.wvalid & self.bus.wready & self.bus.wlast):
                    m.next = "WAIT_B"

            with m.State("WAIT_B"):
                m.d.comb += self.bus.bready.eq(1)
                with m.If(self.bus.bvalid):
                    m.next = "SEND_RESP"

            with m.State("SEND_RESP"):
                total_bytes = beats_per_row * dma_beat_bytes
                lg_size_val = exact_log2(total_bytes)
                m.d.comb += [
                    self.resp_valid.eq(1),
                    self.resp_lg_size.eq(lg_size_val),
                    self.resp_source.eq(wr_source_counter),
                ]
                with m.If(self.resp_ready):
                    m.d.sync += wr_source_counter.eq(wr_source_counter + 1)
                    m.next = "IDLE"

        return m

# ===========================================================================
# ReadBridge tests
# ===========================================================================

@cocotb.test()
async def test_rb_single_beat(dut):
    """Single-beat read: lg_size=4 (16 bytes) → 1 AXI beat."""
    await cocotb_init(dut)

    ADDR = 0x1000
    SOURCE = 3
    LG_SIZE = 4  # 16 bytes = 1 beat at 128-bit bus
    DATA = 0xDEADBEEF_CAFEBABE_12345678_AABBCCDD

    # --- send request (ready is comb high in IDLE, accepted on next edge) ---
    dut.req_valid.value = 1
    dut.req_vaddr.value = ADDR
    dut.req_source.value = SOURCE
    dut.req_lg_size.value = LG_SIZE
    dut.resp_ready.value = 1
    dut.bus__arready.value = 1
    dut.bus__rvalid.value = 0

    await RisingEdge(dut.clk)  # handshake fires here
    dut.req_valid.value = 0

    # --- expect AR ---
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__arvalid.value and dut.bus__arready.value:
            ar_addr = dut.bus__araddr.value.to_unsigned()
            ar_len = dut.bus__arlen.value.to_unsigned()
            break
    else:
        assert False, "AR never fired"
    dut.bus__arready.value = 0

    assert ar_addr == ADDR, f"AR addr: {hex(ar_addr)} != {hex(ADDR)}"
    assert ar_len == 0, f"AR len: {ar_len} != 0 (single beat)"

    # --- provide R data ---
    dut.bus__rvalid.value = 1
    dut.bus__rdata.value = DATA
    dut.bus__rlast.value = 1
    dut.bus__rresp.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.resp_valid.value and dut.resp_ready.value:
            got_data = dut.resp_data.value.to_unsigned()
            got_source = dut.resp_source.value.to_unsigned()
            got_last = int(dut.resp_last.value)
            break
    else:
        assert False, "resp never received"
    dut.bus__rvalid.value = 0

    assert got_data == DATA, f"data: {hex(got_data)} != {hex(DATA)}"
    assert got_source == SOURCE, f"source: {got_source} != {SOURCE}"
    assert got_last == 1, f"last: {got_last} != 1"
    dut._log.info("PASS rb_single_beat")


@cocotb.test()
async def test_rb_multi_beat(dut):
    """Multi-beat read: lg_size=5 (32 bytes) → 2 AXI beats."""
    await cocotb_init(dut)

    ADDR = 0x2000
    SOURCE = 7
    LG_SIZE = 5  # 32 bytes = 2 beats at 128-bit bus
    BEATS = [0x1111111111111111_2222222222222222, 0x3333333333333333_4444444444444444]

    dut.req_valid.value = 1
    dut.req_vaddr.value = ADDR
    dut.req_source.value = SOURCE
    dut.req_lg_size.value = LG_SIZE
    dut.resp_ready.value = 1
    dut.bus__arready.value = 1
    dut.bus__rvalid.value = 0

    await RisingEdge(dut.clk)  # handshake fires
    dut.req_valid.value = 0

    # AR
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__arvalid.value and dut.bus__arready.value:
            ar_len = dut.bus__arlen.value.to_unsigned()
            break
    dut.bus__arready.value = 0
    assert ar_len == 1, f"AR len: {ar_len} != 1 (2 beats)"

    # R beats
    got = []
    got_last = []
    for i, beat_data in enumerate(BEATS):
        dut.bus__rvalid.value = 1
        dut.bus__rdata.value = beat_data
        dut.bus__rlast.value = int(i == len(BEATS) - 1)
        dut.bus__rresp.value = 0
        for _ in range(10):
            await RisingEdge(dut.clk)
            await ReadWrite()
            if dut.resp_valid.value and dut.resp_ready.value:
                got.append(dut.resp_data.value.to_unsigned())
                got_last.append(int(dut.resp_last.value))
                break
        else:
            assert False, f"resp beat {i} never received"
    dut.bus__rvalid.value = 0

    assert got == BEATS, f"data mismatch"
    assert got_last == [0, 1], f"last: {got_last}"
    dut._log.info("PASS rb_multi_beat")


@cocotb.test()
async def test_rb_back_to_back(dut):
    """Two read requests back-to-back."""
    await cocotb_init(dut)

    dut.resp_ready.value = 1
    dut.bus__arready.value = 1
    dut.bus__rvalid.value = 0

    for req_i, (addr, src, data) in enumerate([
        (0x1000, 2, 0xAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDD),
        (0x2000, 5, 0x11111111_22222222_33333333_44444444),
    ]):
        # send req
        dut.req_valid.value = 1
        dut.req_vaddr.value = addr
        dut.req_source.value = src
        dut.req_lg_size.value = 4
        for _ in range(10):
            await RisingEdge(dut.clk)
            await ReadWrite()
            if dut.req_ready.value:
                break
        dut.req_valid.value = 0

        # AR
        for _ in range(10):
            await RisingEdge(dut.clk)
            await ReadWrite()
            if dut.bus__arvalid.value and dut.bus__arready.value:
                assert dut.bus__araddr.value.to_unsigned() == addr
                break

        # R
        dut.bus__rvalid.value = 1
        dut.bus__rdata.value = data
        dut.bus__rlast.value = 1
        dut.bus__rresp.value = 0
        for _ in range(10):
            await RisingEdge(dut.clk)
            await ReadWrite()
            if dut.resp_valid.value and dut.resp_ready.value:
                assert dut.resp_data.value.to_unsigned() == data
                assert dut.resp_source.value.to_unsigned() == src
                break
        dut.bus__rvalid.value = 0

        # wait for bridge to return to IDLE
        for _ in range(5):
            await RisingEdge(dut.clk)

    dut._log.info("PASS rb_back_to_back")


# ===========================================================================
# WriteBridge tests
# ===========================================================================

@cocotb.test()
async def test_wb_single_beat(dut):
    """Single-beat write: one writeReqPacket with last=1."""
    await cocotb_init(dut)

    ADDR = 0x3000
    DATA = 0xDEADBEEF_CAFEBABE_12345678_AABBCCDD

    dut.req_valid.value = 1
    dut.req_vaddr.value = ADDR
    dut.req_data.value = DATA
    dut.req_last.value = 1
    dut.resp_ready.value = 1
    dut.bus__awready.value = 0
    dut.bus__wready.value = 0
    dut.bus__bvalid.value = 0

    # IDLE → SEND_AW: bridge captures vaddr, doesn't ack req yet
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__awvalid.value:
            break

    # accept AW
    dut.bus__awready.value = 1
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__awvalid.value and dut.bus__awready.value:
            aw_addr = dut.bus__awaddr.value.to_unsigned()
            break
    else:
        assert False, "AW never fired"
    dut.bus__awready.value = 0

    assert aw_addr == ADDR, f"AW addr: {hex(aw_addr)} != {hex(ADDR)}"

    # accept W (bridge acks writeReqPacket here)
    dut.bus__wready.value = 1
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__wvalid.value and dut.bus__wready.value:
            w_data = dut.bus__wdata.value.to_unsigned()
            w_last = int(dut.bus__wlast.value)
            break
    else:
        assert False, "W never fired"
    dut.bus__wready.value = 0
    dut.req_valid.value = 0

    assert w_data == (DATA & ((1 << 128) - 1)), f"W data: {hex(w_data)}"
    assert w_last == 1, f"wlast: {w_last}"

    # provide B
    dut.bus__bvalid.value = 1
    dut.bus__bresp.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__bready.value and dut.bus__bvalid.value:
            break
    dut.bus__bvalid.value = 0

    # check writeRespPacket
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.resp_valid.value and dut.resp_ready.value:
            break
    else:
        assert False, "write resp never received"

    dut._log.info("PASS wb_single_beat")


@cocotb.test()
async def test_wb_multi_beat(dut):
    """Multi-beat write: 4 writeReqPacket beats (512 bits total)."""
    await cocotb_init(dut)

    ADDR = 0x4000
    BEATS = [
        (0x11111111_22222222_33333333_44444444, 0),
        (0x55555555_66666666_77777777_88888888, 0),
        (0x99999999_AAAAAAAA_BBBBBBBB_CCCCCCCC, 0),
        (0xDDDDDDDD_EEEEEEEE_FFFFFFFF_00000000, 1),  # last
    ]

    dut.resp_ready.value = 1
    dut.bus__awready.value = 1
    dut.bus__wready.value = 1
    dut.bus__bvalid.value = 0

    # First beat triggers IDLE→SEND_AW
    dut.req_valid.value = 1
    dut.req_vaddr.value = ADDR
    dut.req_data.value = BEATS[0][0]
    dut.req_last.value = BEATS[0][1]

    # Wait for AW
    for _ in range(20):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__awvalid.value and dut.bus__awready.value:
            assert dut.bus__awaddr.value.to_unsigned() == ADDR
            aw_len = dut.bus__awlen.value.to_unsigned()
            break
    else:
        assert False, "AW never fired"

    assert aw_len == 3, f"awlen: {aw_len} != 3 (4 beats)"

    # W beats — bridge acks writeReqPacket in SEND_W state
    got_w = []
    beat_idx = 0
    for _ in range(50):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__wvalid.value and dut.bus__wready.value:
            got_w.append(dut.bus__wdata.value.to_unsigned())
            if dut.bus__wlast.value:
                break
        if dut.req_ready.value and dut.req_valid.value:
            beat_idx += 1
            if beat_idx < len(BEATS):
                dut.req_data.value = BEATS[beat_idx][0]
                dut.req_last.value = BEATS[beat_idx][1]
            else:
                dut.req_valid.value = 0

    expected_w = [(b & ((1 << 128) - 1)) for b, _ in BEATS]
    assert got_w == expected_w, f"W data mismatch: {[hex(w) for w in got_w]}"

    # B
    dut.bus__bvalid.value = 1
    dut.bus__bresp.value = 0
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.bus__bready.value:
            break
    dut.bus__bvalid.value = 0

    # write resp
    for _ in range(10):
        await RisingEdge(dut.clk)
        await ReadWrite()
        if dut.resp_valid.value and dut.resp_ready.value:
            break
    else:
        assert False, "write resp never received"

    dut._log.info("PASS wb_multi_beat")


# ===========================================================================
# Pytest entry points
# ===========================================================================

def test_read_bridge():
    cocotb_run(ReadBridge(), testfilter="test_rb_.*")


def test_write_bridge():
    cocotb_run(WriteBridge(), testfilter="test_wb_.*")
