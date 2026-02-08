import json
import os
from dataclasses import dataclass
from pathlib import Path

import amaranth as am
from amaranth.lib import data, stream
from amaranth.lib.wiring import In, Out, Component, connect, flipped
from amaranth.utils import exact_log2
from amaranth_soc import csr

from gemmini_amaranth.bus import AXI4, BurstType, AXI4Lite, AXI4LiteCSRBridge


@dataclass(frozen=True)
class GemminiConfig:
    dim: int = 16
    dma_buswidth: int = 128
    write_data_width: int = 512
    vaddr_width: int = 32
    source_width: int = 4
    csr_data_width: int = 32
    verilog_dir: str = "verilog"
    csr_offsets = {"status": 0x00}

    @classmethod
    def fromdict(cls, d): return cls(**d)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            j = json.load(f)
        return cls(dim=j["dim"], dma_buswidth=j["dmaBuswidth"],
                   write_data_width=j["writeDataWidth"], vaddr_width=j["vaddrWidth"],
                   source_width=j["readReqPacket"]["sourceWidth"])

    @classmethod
    def generate(cls, *, build_dir=None, chisel_dir=None, verbose=False, **kwargs):
        """Generate Chisel Verilog and return a matching :class:`GemminiConfig`.

        Caches output under ``build_dir/<hash>/`` so repeated calls with the
        same parameters skip regeneration.  Set ``ALWAYS=1`` env var or pass
        ``always=True`` (in *kwargs*) to force rebuild.
        """
        from gemmini_amaranth.generate import build_gen_params, config_hash, generate_verilog

        always = kwargs.pop("always", bool(int(os.environ.get("ALWAYS", "0"))))
        gen_params = build_gen_params(**kwargs)
        cache_key = config_hash(gen_params)

        if build_dir is None:
            build_dir = Path(__file__).resolve().parent.parent / "build"
        output_dir = Path(build_dir) / cache_key
        config_json = output_dir / "gemmini_config.json"

        if always or not config_json.exists():
            generate_verilog(gen_params, output_dir, chisel_dir, verbose)

        cfg = cls.from_json(config_json)
        return cls(dim=cfg.dim, dma_buswidth=cfg.dma_buswidth,
                   write_data_width=cfg.write_data_width, vaddr_width=cfg.vaddr_width,
                   source_width=cfg.source_width, verilog_dir=str(output_dir))

    def get_verilog_sources(self): return sorted(Path(self.verilog_dir).glob("*.v"))


CMD_LAYOUT = data.StructLayout({"funct": 7, "rs1": 64, "rs2": 64, "xd": 1, "rd": 5})
RESP_LAYOUT = data.StructLayout({"rd": 5, "data": 64})


class Gemmini(Component):
    def __init__(self, config=None, **kwargs):
        if config is None and not kwargs:
            config = GemminiConfig()
        elif config is None:
            config = GemminiConfig.generate(**kwargs)
        elif kwargs:
            raise TypeError("Pass either a GemminiConfig or keyword arguments, not both")
        self.config = config
        csrs = csr.Builder(addr_width=1, data_width=config.csr_data_width)
        self.status_reg = csrs.add("status",
            csr.Register({"busy": csr.Field(csr.action.R, 1)}, access="r"),
            offset=config.csr_offsets["status"])
        self.csr_bridge = csr.Bridge(csrs.as_memory_map())
        self.axi4lite_bridge = AXI4LiteCSRBridge(self.csr_bridge.bus)

        super().__init__({
            "bus": Out(AXI4(addr_width=config.vaddr_width, data_width=config.dma_buswidth)),
            "cmd": In(stream.Signature(CMD_LAYOUT)),
            "resp": Out(stream.Signature(RESP_LAYOUT)),
            "ctrl": In(AXI4Lite(addr_width=3, data_width=config.csr_data_width)),
        })

    def elaborate(self, _):
        m = am.Module()
        cfg = self.config
        axi_beat_bytes = cfg.dma_buswidth // 8
        dma_beat_bytes = cfg.dma_buswidth // 8
        beats_per_row = cfg.write_data_width // cfg.dma_buswidth

        m.submodules.csr_bridge = self.csr_bridge
        m.submodules.axi4lite_bridge = self.axi4lite_bridge
        connect(m, flipped(self.ctrl), self.axi4lite_bridge.bus)

        cmd_ready, resp_valid, resp_rd, resp_data, io_busy = (
            am.Signal(), am.Signal(), am.Signal(5), am.Signal(64), am.Signal())
        m.d.comb += self.status_reg.f.busy.r_data.eq(io_busy)

        rr_valid, rr_ready = am.Signal(), am.Signal()
        rr_lg_size, rr_bytes_read, rr_shift = am.Signal(3), am.Signal(7), am.Signal(6)
        rr_vaddr, rr_source = am.Signal(cfg.vaddr_width), am.Signal(cfg.source_width)
        rresp_valid, rresp_ready = am.Signal(), am.Signal()
        rresp_source, rresp_data = am.Signal(cfg.source_width), am.Signal(cfg.dma_buswidth)
        rresp_lg_size, rresp_last = am.Signal(3), am.Signal()
        wr_valid, wr_ready = am.Signal(), am.Signal()
        wr_data, wr_vaddr, wr_last = am.Signal(cfg.write_data_width), am.Signal(cfg.vaddr_width), am.Signal()
        wresp_valid, wresp_ready = am.Signal(), am.Signal()
        wresp_lg_size, wresp_source = am.Signal(3), am.Signal(cfg.source_width)

        m.submodules.gemmini = am.Instance("Gemmini",
            i_clock=am.ClockSignal(), i_reset=am.ResetSignal(),
            o_io_cmd_ready=cmd_ready, i_io_cmd_valid=self.cmd.valid,
            i_io_cmd_bits_inst_funct=self.cmd.payload.funct,
            i_io_cmd_bits_rs1=self.cmd.payload.rs1, i_io_cmd_bits_rs2=self.cmd.payload.rs2,
            i_io_cmd_bits_inst_xd=self.cmd.payload.xd, i_io_cmd_bits_inst_rd=self.cmd.payload.rd,
            i_io_cmd_bits_inst_rs2=0, i_io_cmd_bits_inst_rs1=0,
            i_io_cmd_bits_inst_xs1=1, i_io_cmd_bits_inst_xs2=1,
            i_io_cmd_bits_inst_opcode=0x0B,
            i_io_cmd_bits_status_prv=0b11, i_io_cmd_bits_status_dprv=0b11,
            i_io_cmd_bits_status_mpp=0b11,
            **{f"i_io_cmd_bits_status_{f}": 0 for f in [
                "debug", "cease", "wfi", "isa", "dv", "v", "sd", "zero2", "mpv", "gva",
                "mbe", "sbe", "sxl", "uxl", "sd_rv32", "zero1", "tsr", "tw", "tvm", "mxr",
                "sum", "mprv", "xs", "fs", "vs", "spp", "mpie", "ube", "spie", "upie",
                "mie", "hie", "sie", "uie"]},
            i_io_resp_ready=self.resp.ready, o_io_resp_valid=resp_valid,
            o_io_resp_bits_rd=resp_rd, o_io_resp_bits_data=resp_data,
            o_io_busy=io_busy,
            i_io_readReqPacket_ready=rr_ready, o_io_readReqPacket_valid=rr_valid,
            o_io_readReqPacket_bits_lg_size=rr_lg_size,
            o_io_readReqPacket_bits_bytes_read=rr_bytes_read,
            o_io_readReqPacket_bits_shift=rr_shift,
            o_io_readReqPacket_bits_vaddr=rr_vaddr,
            o_io_readReqPacket_bits_source=rr_source,
            o_io_readRespPacket_ready=rresp_ready,
            i_io_readRespPacket_valid=rresp_valid,
            i_io_readRespPacket_bits_source=rresp_source,
            i_io_readRespPacket_bits_data=rresp_data,
            i_io_readRespPacket_bits_lg_size=rresp_lg_size,
            i_io_readRespPacket_bits_last=rresp_last,
            i_io_writeReqPacket_ready=wr_ready, o_io_writeReqPacket_valid=wr_valid,
            o_io_writeReqPacket_bits_data=wr_data,
            o_io_writeReqPacket_bits_vaddr=wr_vaddr,
            o_io_writeReqPacket_bits_last=wr_last,
            o_io_writeRespPacket_ready=wresp_ready,
            i_io_writeRespPacket_valid=wresp_valid,
            i_io_writeRespPacket_bits_lg_size=wresp_lg_size,
            i_io_writeRespPacket_bits_source=wresp_source,
        )

        m.d.comb += [
            self.cmd.ready.eq(cmd_ready), self.resp.valid.eq(resp_valid),
            self.resp.payload.rd.eq(resp_rd), self.resp.payload.data.eq(resp_data),
        ]

        # read bridge: readReqPacket → AXI AR/R → readRespPacket
        rd_src_r, rd_lg_r, rd_va_r, rd_arlen_r = (
            am.Signal(cfg.source_width), am.Signal(3), am.Signal(cfg.vaddr_width), am.Signal(8))
        m.d.comb += [self.bus.arburst.eq(BurstType.INCR), self.bus.arsize.eq(exact_log2(axi_beat_bytes))]

        with m.FSM(name="read_bridge"):
            with m.State("IDLE"):
                m.d.comb += rr_ready.eq(1)
                with m.If(rr_valid):
                    m.d.sync += [rd_src_r.eq(rr_source), rd_lg_r.eq(rr_lg_size), rd_va_r.eq(rr_vaddr)]
                    for lg in range(8):
                        with m.If(rr_lg_size == lg):
                            m.d.sync += rd_arlen_r.eq(max(1, (1 << lg) // dma_beat_bytes) - 1)
                    m.next = "SEND_AR"
            with m.State("SEND_AR"):
                m.d.comb += [self.bus.araddr.eq(rd_va_r), self.bus.arlen.eq(rd_arlen_r), self.bus.arvalid.eq(1)]
                with m.If(self.bus.arready): m.next = "RECV_R"
            with m.State("RECV_R"):
                m.d.comb += [
                    rresp_valid.eq(self.bus.rvalid), rresp_data.eq(self.bus.rdata),
                    rresp_source.eq(rd_src_r), rresp_lg_size.eq(rd_lg_r),
                    rresp_last.eq(self.bus.rlast), self.bus.rready.eq(rresp_ready)]
                with m.If(self.bus.rvalid & self.bus.rready & self.bus.rlast): m.next = "DONE"
            with m.State("DONE"): m.next = "IDLE"

        # write bridge: writeReqPacket → AXI AW/W/B → writeRespPacket
        wr_va_r = am.Signal(cfg.vaddr_width)
        m.d.comb += [self.bus.awburst.eq(BurstType.INCR), self.bus.awsize.eq(exact_log2(axi_beat_bytes))]

        with m.FSM(name="write_bridge"):
            with m.State("IDLE"):
                m.d.comb += wr_ready.eq(0)
                with m.If(wr_valid):
                    m.d.sync += wr_va_r.eq(wr_vaddr)
                    m.next = "SEND_AW"
            with m.State("SEND_AW"):
                m.d.comb += [self.bus.awaddr.eq(wr_va_r), self.bus.awlen.eq(beats_per_row - 1),
                             self.bus.awvalid.eq(1), wr_ready.eq(0)]
                with m.If(self.bus.awready): m.next = "SEND_W"
            with m.State("SEND_W"):
                m.d.comb += [self.bus.wdata.eq(wr_data[:cfg.dma_buswidth]), self.bus.wvalid.eq(wr_valid),
                             self.bus.wlast.eq(wr_last), wr_ready.eq(self.bus.wready)]
                with m.If(self.bus.wvalid & self.bus.wready & self.bus.wlast): m.next = "WAIT_B"
            with m.State("WAIT_B"):
                m.d.comb += self.bus.bready.eq(1)
                with m.If(self.bus.bvalid): m.next = "SEND_RESP"
            with m.State("SEND_RESP"):
                m.d.comb += [wresp_valid.eq(1), wresp_lg_size.eq(exact_log2(beats_per_row * dma_beat_bytes)),
                             wresp_source.eq(0)]
                with m.If(wresp_ready): m.next = "IDLE"

        return m


if __name__ == "__main__":
    from amaranth.back import verilog
    config = GemminiConfig()
    g = Gemmini(config)
    with open("gemmini.v", "w") as f:
        f.write(verilog.convert(g, name="Gemmini", emit_src=False))
    print("Generated gemmini.v")
