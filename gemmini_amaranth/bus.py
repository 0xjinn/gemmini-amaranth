"""AXI4 bus primitives. Should eventually live in amaranth-soc."""

from enum import Enum

import amaranth as am
from amaranth.lib.wiring import In, Out, Component
from amaranth.lib import wiring, stream, data
from amaranth.utils import exact_log2


class BurstType(Enum):
    FIXED = 0b00
    INCR  = 0b01
    WRAP  = 0b10


class RespType(Enum):
    OKAY   = 0b00
    EXOKAY = 0b01
    SLVERR = 0b10
    DECERR = 0b11


class AXI4Lite(wiring.Signature):
    def __init__(self, *, addr_width, data_width):
        assert data_width % 8 == 0
        super().__init__({
            "awaddr":  Out(addr_width),
            "awprot":  Out(3),
            "awvalid": Out(1),
            "awready": In(1),

            "wdata":   Out(data_width),
            "wstrb":   Out(data_width // 8),
            "wvalid":  Out(1),
            "wready":  In(1),

            "bresp":   In(RespType),
            "bvalid":  In(1),
            "bready":  Out(1),

            "araddr":  Out(addr_width),
            "arprot":  Out(3),
            "arvalid": Out(1),
            "arready": In(1),

            "rdata":   In(data_width),
            "rresp":   In(RespType),
            "rvalid":  In(1),
            "rready":  Out(1),
        })


class AXI4LiteCSRBridge(wiring.Component):
    def __init__(self, csr_bus, *, data_width=None):
        assert csr_bus.data_width == 32
        self.csr_bus = csr_bus
        data_width = csr_bus.data_width if data_width is None else data_width

        axi4lite_sig = AXI4Lite(addr_width=csr_bus.addr_width + 2, data_width=data_width)
        super().__init__({"bus": In(axi4lite_sig)})

    def elaborate(self, _):
        m = am.Module()

        aw_hs = self.bus.awvalid & self.bus.awready
        w_hs = self.bus.wvalid & self.bus.wready
        b_hs = self.bus.bvalid & self.bus.bready
        ar_hs = self.bus.arvalid & self.bus.arready
        r_hs = self.bus.rvalid & self.bus.rready

        write_ready = self.bus.awvalid & self.bus.wvalid & (~self.bus.bvalid | self.bus.bready)

        m.d.comb += [
            self.bus.awready.eq(write_ready),
            self.bus.wready.eq(write_ready),
            self.csr_bus.w_stb.eq(aw_hs & w_hs),
            self.csr_bus.w_data.eq(self.bus.wdata),
            self.csr_bus.addr.eq(am.Mux(self.csr_bus.w_stb, self.bus.awaddr >> 2, self.bus.araddr >> 2)),
        ]

        with m.If(self.csr_bus.w_stb):
            m.d.sync += self.bus.bvalid.eq(1)
        with m.Elif(b_hs):
            m.d.sync += self.bus.bvalid.eq(0)

        rdata_q = am.Signal.like(self.csr_bus.r_data)
        rvalid_q = am.Signal()
        with m.If(self.bus.rvalid & ~self.bus.rready & ~rvalid_q):
            m.d.sync += [rvalid_q.eq(1), rdata_q.eq(self.csr_bus.r_data)]
        with m.Elif(rvalid_q & self.bus.rready):
            m.d.sync += rvalid_q.eq(0)

        with m.If(self.csr_bus.r_stb):
            m.d.sync += self.bus.rvalid.eq(1)
        with m.Elif(r_hs):
            m.d.sync += self.bus.rvalid.eq(0)

        m.d.comb += [
            self.csr_bus.r_stb.eq(ar_hs),
            self.bus.rdata.eq(am.Mux(rvalid_q, rdata_q, self.csr_bus.r_data)),
            self.bus.arready.eq((~self.bus.rvalid | self.bus.rready) & ~self.csr_bus.w_stb),
        ]
        return m


class AXI4(wiring.Signature):
    def __init__(self, *, addr_width, data_width):
        if data_width % 8 != 0 or data_width < 8 or data_width > 1024:
            raise ValueError(f"invalid data_width: {data_width}")
        self.addr_width = addr_width
        self.data_width = data_width

        super().__init__({
            "awid":     Out(1),
            "awaddr":   Out(addr_width),
            "awlen":    Out(8),
            "awsize":   Out(3, init=exact_log2(data_width // 8)),
            "awburst":  Out(BurstType, init=BurstType.INCR),
            "awlock":   Out(1),
            "awcache":  Out(4),
            "awprot":   Out(3),
            "awvalid":  Out(1),
            "awready":  In(1),
            "awqos":    Out(4),

            "wid":      Out(1),
            "wdata":    Out(data_width),
            "wstrb":    Out(data_width // 8, init=(1 << data_width // 8) - 1),
            "wlast":    Out(1),
            "wvalid":   Out(1),
            "wready":   In(1),

            "bid":      In(1),
            "bresp":    In(RespType),
            "bvalid":   In(1),
            "bready":   Out(1),

            "arid":     Out(1),
            "araddr":   Out(addr_width),
            "arlen":    Out(8),
            "arsize":   Out(3, init=exact_log2(data_width // 8)),
            "arburst":  Out(BurstType, init=BurstType.INCR),
            "arlock":   Out(1),
            "arcache":  Out(4),
            "arprot":   Out(3),
            "arvalid":  Out(1),
            "arready":  In(1),
            "arqos":    Out(4),

            "rid":      In(1),
            "rdata":    In(data_width),
            "rresp":    In(RespType),
            "rlast":    In(1),
            "rvalid":   In(1),
            "rready":   Out(1),
        })


class Serializer(Component):
    def __init__(self, *, src_width, dst_width):
        self.nbeats = -(-src_width // dst_width)
        self.pisoreg = am.Signal(data.ArrayLayout(dst_width, self.nbeats))
        self.beat_counter = am.Signal(range(self.nbeats + 1))

        super().__init__({
            "src": In(stream.Signature(data.StructLayout({"data": src_width, "last": 1}))),
            "dst": Out(stream.Signature(data.StructLayout({"data": dst_width, "last": 1}))),
        })

    def elaborate(self, _):
        m = am.Module()

        last_q = am.Signal()
        done = self.beat_counter == 0
        last_beat = self.beat_counter == 1

        with m.If(self.src.valid & self.src.ready):
            m.d.sync += [
                self.beat_counter.eq(self.nbeats),
                self.pisoreg.eq(self.src.payload.data),
                last_q.eq(self.src.payload.last),
            ]
        with m.Elif(~done & self.dst.ready):
            m.d.sync += [
                self.beat_counter.eq(self.beat_counter - 1),
                self.pisoreg.eq(self.pisoreg[1:]),
            ]
            with m.If(last_q & last_beat):
                m.d.sync += last_q.eq(0)

        m.d.comb += [
            self.src.ready.eq(done),
            self.dst.valid.eq(~done),
            self.dst.payload.data.eq(self.pisoreg[0]),
            self.dst.payload.last.eq(last_q & last_beat),
        ]
        return m


class Deserializer(Component):
    def __init__(self, *, src_width, dst_width):
        self.nbeats = -(-dst_width // src_width)
        self.siporeg = am.Signal(data.ArrayLayout(src_width, self.nbeats))
        self.beat_counter = am.Signal(range(self.nbeats + 1))
        super().__init__({"src": In(stream.Signature(src_width)), "dst": Out(stream.Signature(dst_width))})

    def elaborate(self, _):
        m = am.Module()

        done = self.beat_counter == self.nbeats

        with m.If(self.src.valid & self.src.ready):
            m.d.sync += [
                self.beat_counter.eq(am.Mux(done, 1, self.beat_counter + 1)),
                self.siporeg.eq(am.Cat(self.siporeg[1:], self.src.payload)),
            ]
        with m.Elif(self.dst.valid & self.dst.ready):
            m.d.sync += self.beat_counter.eq(0)

        m.d.comb += [
            self.src.ready.eq(~done | self.dst.ready),
            self.dst.valid.eq(done),
            self.dst.payload.eq(self.siporeg),
        ]
        return m
